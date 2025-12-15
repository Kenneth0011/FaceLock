import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import compute_score
import pdb

# --- Matplotlib 設定 (防止在無顯示器環境報錯) ---
import os
os.environ['MPLBACKEND'] = 'Agg' 
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------
# 繪圖輔助函數
# -----------------------------
def plot_facelock_history(history, save_name="facelock_robust_convergence.png"):
    """
    繪製 Loss 收斂曲線
    """
    print("Plotting losses...")

    plt.figure(figsize=(12, 10))
    plt.suptitle('FaceLock Robust Optimization History', fontsize=16)

    # 圖 1: Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 圖 2: CVL Loss (人臉辨識分數)
    plt.subplot(2, 2, 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (High = Preserved)')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)

    # 圖 3: Encoder Loss (潛在空間距離)
    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (Latent Consistency)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 圖 4: LPIPS Loss (影像感知相似度)
    plt.subplot(2, 2, 4)
    plt.plot(history['loss_lpips'], color='purple')
    plt.title('Perceptual Similarity (LPIPS)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_name) 
    print(f"Loss plot saved to: {save_name}")
    plt.close()

# -----------------------------
# 核心函數: FaceLock Robust
# -----------------------------
def facelock_robust(X, model, aligner, fr_model, lpips_fn, 
                    eps=0.03, step_size=0.01, iters=100, 
                    clamp_min=-1, clamp_max=1, 
                    decay=1.0,       # [新增] 動量衰減係數 (1.0 = 累積所有歷史梯度)
                    noise_std=0.005, # [新增] 模擬背景編輯的干擾雜訊強度
                    plot_history=False): 
    """
    FaceLock 的魯棒性增強版 (Robust Version)
    針對背景編輯 (Diffusion Inpainting) 進行優化：
    1. 引入 Momentum (MI-FGSM) 避免陷入局部極值，增加特徵穿透力。
    2. 引入 Noise Injection 模擬編輯過程中的重採樣干擾。
    """
    
    # 1. 初始化對抗樣本
    # 保持與原始輸入相同的裝置和型態 (half/float)
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max)
    
    # 如果原圖是 float16 (half)，則轉為 float32 進行梯度計算以求精確，最後再轉回
    is_half = (X.dtype == torch.float16)
    if is_half:
        X_adv = X_adv.float()
        X = X.float()
    
    X_adv.requires_grad_(True)
    
    vae = model.vae
    # 鎖定原始圖像的 Latent Code 作為基準
    with torch.no_grad():
        clean_latent = vae.encode(X).latent_dist.mean.detach()

    # [新增] 初始化動量 (Momentum Buffer)
    momentum = torch.zeros_like(X_adv).detach().to(X.device)

    history = {
        'total_loss': [],
        'loss_cvl': [],
        'loss_encoder': [],
        'loss_lpips': []
    }

    print(f"Starting FaceLock Robust Attack (Iters={iters}, Momentum={decay}, Noise={noise_std})...")
    pbar = tqdm(range(iters))
    
    for i in pbar:
        # 動態調整 step size (線性衰減)
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        # --- Forward Pass ---
        # 取得 Latent 並重建圖像
        latent = vae.encode(X_adv).latent_dist.mean
        image_rec = vae.decode(latent).sample.clip(-1, 1)

        # [關鍵優化] 魯棒性增強 (Robustness Augmentation)
        # 在計算 FR Score 前加入隨機雜訊，強迫防禦特徵在干擾下仍能保持
        aug_noise = torch.randn_like(image_rec) * noise_std
        image_noisy = image_rec + aug_noise
        
        # --- Loss 計算 ---
        # 1. Identity Loss (CVL): 越高代表身分越保留。
        #    我們希望保留身分 (FaceLock)，所以通常是 Maximize Score => Minimize (-Score)
        loss_cvl = compute_score(image_noisy, X, aligner=aligner, fr_model=fr_model)
        
        # 2. Encoder Loss: 確保 Latent 沒有偏離太遠 (維持結構)
        loss_encoder = F.mse_loss(latent, clean_latent)
        
        # 3. LPIPS Loss: 確保視覺上與原圖相似
        loss_lpips = lpips_fn(image_rec, X) 
        
        # --- 權重排程 (Loss Scheduling) ---
        # 為了抵抗背景指令，我們讓 CVL (身分鎖定) 的權重更早介入
        w_cvl = 2.0 if i >= iters * 0.15 else 0.0  # 提早介入 (原為 0.35 -> 改為 0.15)
        w_lpips = 1.0 if i > iters * 0.25 else 0.0
        
        # 總 Loss
        loss = -loss_cvl * w_cvl + loss_encoder * 0.2 + loss_lpips * w_lpips
        
        # --- Backward Pass ---
        if X_adv.grad is not None:
            X_adv.grad.zero_()
            
        grad, = torch.autograd.grad(loss, [X_adv])
        
        # --- [關鍵優化] 動量更新 (Momentum Update) ---
        # 計算梯度的 L1 Norm 以穩定更新幅度
        grad_norm = torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad / (grad_norm + 1e-10) # 避免除以 0
        
        # 更新動量： momentum = decay * momentum + grad
        momentum = decay * momentum + grad
        
        # 使用動量的符號進行更新 (MI-FGSM)
        X_adv = X_adv - momentum.sign() * actual_step_size # 注意方向：我們要 Minimize Loss

        # --- 投影與截斷 (Projection & Clamping) ---
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        
        # 清空梯度以節省記憶體
        X_adv.grad = None

        # --- 記錄數據 ---
        pbar.set_postfix(
            cvl=f"{loss_cvl.item():.4f}", 
            enc=f"{loss_encoder.item():.4f}", 
            lpips=f"{loss_lpips.item():.4f}", 
            loss=f"{loss.item():.4f}"
        )

        history['total_loss'].append(loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())
        history['loss_lpips'].append(loss_lpips.item())

    # 轉回原本的精度 (如果是 half)
    if is_half:
        X_adv = X_adv.half()

    if plot_history:
        plot_facelock_history(history)

    return X_adv, history


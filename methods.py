import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# 假設你的 utils.py 裡面有這個函式
from utils import compute_score 
import pdb

# --- Matplotlib 終極解決方案 (避免視窗報錯) ---
os.environ['MPLBACKEND'] = 'Agg' 
import matplotlib
import matplotlib.pyplot as plt
# -----------------------------

# ==========================================
# [修正] DIM (Diverse Inputs Method)
# ==========================================
def input_diversity(x, resize_rate=0.9, diversity_prob=0.7):
    """
    DIM 實作：隨機縮放與補邊，增加攻擊強健性 (Robustness)
    """
    # 1. 機率性跳過
    if torch.rand(1) > diversity_prob:
        return x

    # 2. [修正] 強制轉為 float32 進行插值運算，避免 float16 在 interpolate 產生 NaN
    orig_dtype = x.dtype
    x = x.float()

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    rnd = torch.randint(low=img_resize, high=img_size, size=(1,)).item()
    
    # 3. 隨機縮放
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    
    # 4. 計算 Padding
    h_rem = img_size - rnd
    w_rem = img_size - rnd
    
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,)).item()
    pad_right = w_rem - pad_left
    
    # 5. 補邊 (補黑色 0)
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    
    # 6. 轉回原本的精度 (如 float16)
    return padded.to(dtype=orig_dtype)


# --- 繪圖輔助函數 ---
def plot_facelock_history(history):
    print("Plotting history...")
    
    has_lpips = 'loss_lpips' in history and len(history['loss_lpips']) > 0
    
    if has_lpips:
        plt.figure(figsize=(12, 10))
        layout = (2, 2)
    else:
        plt.figure(figsize=(18, 5))
        layout = (1, 3)

    plt.suptitle('FaceLock History (Gradient Ascent Mode)', fontsize=16)

    # 圖 1: Objective Function (越高越好)
    plt.subplot(layout[0], layout[1], 1)
    plt.plot(history['total_objective'])
    plt.title('Total Objective (Maximize)')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)

    # 圖 2: Face Distance (越高越好)
    plt.subplot(layout[0], layout[1], 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Distance (Higher = Better Privacy)')
    plt.grid(True)

    # 圖 3: Encoder MSE (越低越好)
    plt.subplot(layout[0], layout[1], 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (Lower = Better Quality)')
    plt.grid(True)

    # 圖 4: LPIPS (如果有)
    if has_lpips:
        plt.subplot(2, 2, 4)
        plt.plot(history['loss_lpips'], color='purple')
        plt.title('Perceptual Loss')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "facelock_ascent_history.png"
    plt.savefig(save_path) 
    print(f"Plot saved to: {save_path}")
    plt.close()


# ==========================================
# [修正] CW L2 attack (Minimization)
# ==========================================
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    """
    CW Attack 是一個最佳化問題 (Minimization)，所以維持梯度下降。
    修正了 Tanh 範圍與 Device 問題。
    """
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean.detach()

    def f(x):
        latents = encoder(x).latent_dist.mean
        # 計算特徵距離
        dist = torch.norm(latents - clean_latents, p=2, dim=-1)
        if targeted:
            # Targeted logic (Minimize distance to target) - 暫略
            return latents.norm() 
        else:
            # Untargeted: Maximize distance => Minimize negative distance
            return -dist
    
    # [修正] 使用 .to(X.device) 
    w = torch.zeros_like(X, requires_grad=True).to(X.device)
    
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    history = {'total_loss': [], 'loss1': [], 'loss2': []} 

    for step in pbar:
        # [修正] Tanh 直接映射到 (-1, 1)，符合 VAE 輸入範圍
        a = nn.Tanh()(w)

        loss1 = nn.MSELoss(reduction='sum')(a, X) # 距離原圖越近越好
        loss2 = torch.sum(c * f(a))               # 攻擊成功率

        cost = loss1 + loss2
        pbar.set_description(f"CW Loss: {cost.item():.5f}")
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        history['total_loss'].append(cost.item())
        history['loss1'].append(loss1.item())
        history['loss2'].append(loss2.item())
        
    # [修正] 回傳正確範圍的圖片
    X_adv = nn.Tanh()(w)
    return X_adv, history 


# ==========================================
# [彈性] Encoder Attack (支援梯度上升)
# ==========================================
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    """
    若是 Untargeted (躲避)，使用梯度上升 (Gradient Ascent) 最大化 MSE。
    若是 Targeted (模仿)，使用梯度下降 (Gradient Descent) 最小化 MSE。
    """
    encoder = model.vae.encode
    
    # 初始化雜訊
    noise = (torch.rand_like(X) * 2 * eps - eps).to(X.device)
    if X.dtype == torch.float16:
        noise = noise.half()
        
    X_adv = torch.clamp(X.clone().detach() + noise, min=clamp_min, max=clamp_max)
    
    clean_latent = encoder(X).latent_dist.mean.detach()
    loss_fn = nn.MSELoss() 
    
    pbar = tqdm(range(iters))
    history = {'loss': []} 
    
    for i in pbar:
        # Decay step size (Optional)
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        
        if targeted:
            # Targeted: Minimize Loss
            # 這裡假設你有個 target_latent，暫時用 norm 代替
            loss = latent.norm() 
            grad, = torch.autograd.grad(loss, [X_adv])
            # Descent: 減去梯度
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            # Untargeted: Maximize MSE (遠離原圖特徵)
            loss = loss_fn(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            # Ascent: 加上梯度
            X_adv = X_adv + grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Encoder Attack]: Loss {loss.item():.5f}")

        # Projection (限制在 Epsilon 球內)
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        
        history['loss'].append(loss.item()) 

    return X_adv, history 


# ==========================================
# VAE Attack (標準版)
# ==========================================
def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    vae = model.vae
    noise = (torch.rand_like(X) * 2 * eps - eps).to(X.device)
    if X.dtype == torch.float16: noise = noise.half()
    
    X_adv = torch.clamp(X.clone().detach() + noise, min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    history = {'loss': []} 
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        # 這裡假設我們要破壞重建品質 (Maximize Error?) 
        # 或者是 Minimize image norm? 依你原本邏輯保留 Minimize
        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        
        # Descent
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[VAE Atk]: Loss {loss.item():.5f}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        history['loss'].append(loss.item()) 

    return X_adv, history 


# ==============================================================
# [重構] FaceLock (Gradient Ascent + DIM)
# ==============================================================
def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.07, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, plot_history=False, tv_weight=0): 
    
    # 1. 初始化雜訊
    noise = (torch.rand_like(X) * 2 * eps - eps).to(X.device)
    if X.dtype == torch.float16:
        noise = noise.half()

    X_adv = X.clone().detach() + noise
    X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
    
    if X.dtype == torch.float16:
        X_adv = X_adv.half()
        
    X_adv.requires_grad = True

    vae = model.vae
    # 取得原始 Latent (基準)
    clean_latent = vae.encode(X).latent_dist.mean.detach()
    if X_adv.dtype == torch.float16:
        clean_latent = clean_latent.half()

    # 記錄 Objective 而非 Loss
    history = {'total_objective': [], 'loss_cvl': [], 'loss_encoder': []}
    pbar = tqdm(range(iters))

    print(f"啟動 Facelock 梯度上升模式 (Gradient Ascent): eps={eps}")

    for i in pbar:
        # 2. Forward Pass
        latent = vae.encode(X_adv).latent_dist.mean
        image_recon = vae.decode(latent).sample.clip(-1, 1)

        # 3. DIM 變形 (增強攻擊遷移性)
        image_dim = input_diversity(image_recon, resize_rate=0.9, diversity_prob=0.7)

        # 4. 計算各項數值
        # (A) 人臉距離 (Distance)：我們希望這個越大越好 (無法辨識)
        # 注意：compute_score 必須回傳距離。如果回傳相似度，請在 Objective 加上負號。
        loss_cvl = compute_score(image_dim.float(), X.float(), aligner=aligner, fr_model=fr_model)
        
        # (B) 結構破壞 (MSE)：我們希望這個越小越好 (畫質正常)
        loss_encoder = F.mse_loss(latent, clean_latent)
        
        # 5. 定義目標函數 (Objective Function) - 我們要 Maximize 這個值
        # Objective = (推遠人臉距離) - (壓低結構破壞)
        objective = (loss_cvl * 5.0) - (loss_encoder * 1.0)

        # 6. 計算梯度
        grad, = torch.autograd.grad(objective, [X_adv])
        
        # 7. 梯度上升更新 (Gradient Ascent: +=)
        X_adv.data = X_adv.data + step_size * grad.sign()

        # 8. 限制範圍 (Projection)
        X_adv.data = torch.max(torch.min(X_adv.data, X + eps), X - eps)
        X_adv.data = torch.clamp(X_adv.data, min=clamp_min, max=clamp_max)

        # 記錄數據
        pbar.set_postfix(obj=f"{objective.item():.3f}", dist=f"{loss_cvl.item():.3f}", mse=f"{loss_encoder.item():.3f}")
        history['total_objective'].append(objective.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())

    # 繪製圖表
    if plot_history:
        try:
            plot_facelock_history(history)
        except Exception as e:
            print(f"Plotting failed: {e}")

    return X_adv, history

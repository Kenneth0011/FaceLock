import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import compute_score
import pdb

# --- Matplotlib 終極解決方案 ---
import os
os.environ['MPLBACKEND'] = 'Agg'  # <-- 關鍵：在 import matplotlib 之前，強制設定環境變數
import matplotlib
import matplotlib.pyplot as plt
# -----------------------------

# ==========================================
# [新增] DIM (Diverse Inputs Method) 輔助函式
# ==========================================
def input_diversity(x, resize_rate=0.9, diversity_prob=0.7):
    """
    DIM 實作：隨機縮放與補邊，增加攻擊強健性
    """
    # 1. 決定是否執行 DIM (有 1-diversity_prob 的機率不執行，直接回傳原圖)
    if torch.rand(1) > diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    # 2. 隨機決定縮放後的大小
    rnd = torch.randint(low=img_resize, high=img_size, size=(1,)).item()
    
    # 3. 縮放圖片
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    
    # 4. 計算需要補邊 (Padding) 的大小
    h_rem = img_size - rnd
    w_rem = img_size - rnd
    
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,)).item()
    pad_right = w_rem - pad_left
    
    # 5. 補邊回原本大小，並補 0 (黑色)
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    
    return padded
# ==========================================


# --- 繪圖輔助函數 ---
def plot_facelock_history(history):
    print("Plotting losses...")
    
    # 檢查是否有 LPIPS 數據，決定要畫幾張圖
    has_lpips = 'loss_lpips' in history and len(history['loss_lpips']) > 0
    
    if has_lpips:
        # 如果有 4 個數據，維持原本的 2x2 版面
        plt.figure(figsize=(12, 10))
        layout = (2, 2)
    else:
        # 如果只有 3 個數據 (極致模式)，改用 1x3 版面
        plt.figure(figsize=(18, 5))
        layout = (1, 3)

    plt.suptitle('FaceLock Loss Convergence (Aggressive + DIM Mode)', fontsize=16)

    # 圖 1: Total Loss
    plt.subplot(layout[0], layout[1], 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 圖 2: CVL Loss (人臉辨識分數 - 越低越好)
    plt.subplot(layout[0], layout[1], 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (loss_cvl)')
    plt.xlabel('Iteration')
    plt.ylabel('Score (Lower is Better)')
    plt.grid(True)

    # 圖 3: Encoder Loss (特徵破壞程度 - 越高越好)
    plt.subplot(layout[0], layout[1], 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (Disruption)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Higher is Better)')
    plt.grid(True)

    # 圖 4: LPIPS (如果有才畫)
    if has_lpips:
        plt.subplot(2, 2, 4)
        plt.plot(history['loss_lpips'], color='purple')
        plt.title('Perceptual Similarity (loss_lpips)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "facelock_loss_convergence.png"
    plt.savefig(save_path) 
    print(f"Loss plot saved to: {save_path}")
    plt.close() # 關閉圖表釋放記憶體
# -----------------------------


# CW L2 attack
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean

    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted:
            return latents.norm()
        else:
            return -torch.norm(latents - clean_latents.detach(), p=2, dim=-1)
    
    w = torch.zeros_like(X, requires_grad=True).cuda()
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    history = {'total_loss': [], 'loss1': [], 'loss2': []} 

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        history['total_loss'].append(cost.item())
        history['loss1'].append(loss1.item())
        history['loss2'].append(loss2.item())
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv, history 

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean
    pbar = tqdm(range(iters))
    
    history = {'loss': []} 
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        if targeted:
            loss = latent.norm()
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = loss_fn(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(norm_2=(X_adv - X).norm().item(), norm_inf=(X_adv - X).abs().max().item())

        history['loss'].append(loss.item()) 

    return X_adv, history 

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    
    history = {'loss': []} 
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        
        history['loss'].append(loss.item()) 

    return X_adv, history 


# ==============================================================
# [修改後] 整合了 DIM (Diverse Inputs Method) 的 Facelock 函式
# ==============================================================
def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.07, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, plot_history=False, tv_weight=0): 
    # [設定] 極致攻擊模式 + DIM
    
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device)
    X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max).half()
    X_adv.requires_grad = True

    vae = model.vae
    clean_latent = vae.encode(X).latent_dist.mean.detach()

    history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': []}
    pbar = tqdm(range(iters))

    print(f"啟動極致隱私模式 (Aggressive Privacy with DIM): eps={eps}")

    for i in pbar:
        # 1. 取得當前圖片特徵與重建圖
        latent = vae.encode(X_adv).latent_dist.mean
        image_recon = vae.decode(latent).sample.clip(-1, 1)

        # ================= [新增] DIM 隨機變形 =================
        # 對要算分數的圖片進行隨機縮放，增加泛化能力
        image_dim = input_diversity(image_recon, resize_rate=0.9, diversity_prob=0.7)
        # ======================================================

        # 2. 計算 Loss
        # (A) 人臉辨識分數 (使用變形後的 image_dim 計算)
        # 這能模擬 "如果圖片被縮放或上傳到FB被壓縮，是否還能防禦成功?"
        loss_cvl = compute_score(image_dim.float(), X.float(), aligner=aligner, fr_model=fr_model)
        
        # (B) 潛在空間距離 (這不需要 DIM，我們針對原始特徵結構攻擊)
        loss_encoder = F.mse_loss(latent, clean_latent)
        
        # 權重分配 (維持原本設定)
        loss = -loss_cvl * 5.0 + loss_encoder * 1.0

        grad, = torch.autograd.grad(loss, [X_adv])
        
        # 3. 更新圖片
        X_adv.data = X_adv.data + step_size * grad.sign()

        # 4. 限制範圍
        X_adv.data = torch.max(torch.min(X_adv.data, X + eps), X - eps)
        X_adv.data = torch.clamp(X_adv.data, min=clamp_min, max=clamp_max)

        # 記錄
        pbar.set_postfix(cvl=f"{loss_cvl.item():.3f}", enc=f"{loss_encoder.item():.3f}")
        history['total_loss'].append(loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())

    if plot_history:
        try:
            plot_facelock_history(history)
        except:
            pass

    return X_adv, history

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
# 1. CW L2 Attack (恢復此函數)
# -----------------------------
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

# -----------------------------
# 2. Encoder Attack (恢復此函數)
# -----------------------------
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

# -----------------------------
# 3. VAE Attack (恢復此函數)
# -----------------------------
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

# -----------------------------
# 4. FaceLock Robust (您的新函數)
# -----------------------------
def facelock_robust(X, model, aligner, fr_model, lpips_fn, 
                    eps=0.03, step_size=0.01, iters=100, 
                    clamp_min=-1, clamp_max=1, 
                    decay=1.0,       # 動量係數
                    noise_std=0.005, # 抗噪強度
                    plot_history=False): 
    
    # 初始化對抗樣本
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max)
    
    # 處理精度問題
    is_half = (X.dtype == torch.float16)
    if is_half:
        X_adv = X_adv.float()
        X = X.float()
    
    X_adv.requires_grad_(True)
    
    vae = model.vae
    with torch.no_grad():
        clean_latent = vae.encode(X).latent_dist.mean.detach()

    # 初始化動量
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
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        # Forward
        latent = vae.encode(X_adv).latent_dist.mean
        image_rec = vae.decode(latent).sample.clip(-1, 1)

        # Robustness Augmentation (Noise Injection)
        aug_noise = torch.randn_like(image_rec) * noise_std
        image_noisy = image_rec + aug_noise
        
        # Loss Calculation
        loss_cvl = compute_score(image_noisy, X, aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        loss_lpips = lpips_fn(image_rec, X) 
        
        # Scheduling
        w_cvl = 2.0 if i >= iters * 0.15 else 0.0
        w_lpips = 1.0 if i > iters * 0.25 else 0.0
        
        loss = -loss_cvl * w_cvl + loss_encoder * 0.2 + loss_lpips * w_lpips
        
        # Backward
        if X_adv.grad is not None:
            X_adv.grad.zero_()
        grad, = torch.autograd.grad(loss, [X_adv])
        
        # Momentum Update
        grad_norm = torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad / (grad_norm + 1e-10)
        momentum = decay * momentum + grad
        
        # Gradient Ascent/Descent
        X_adv = X_adv - momentum.sign() * actual_step_size

        # Projection
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        # Logging
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

    if is_half:
        X_adv = X_adv.half()

    if plot_history:
        plot_facelock_history(history)

    return X_adv, history

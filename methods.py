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

# --- 繪圖輔助函數 (維持存檔) ---
def plot_facelock_history(history):
    """
    專門用來繪製 facelock 函數回傳的 history 字典
    """
    print("Plotting losses...")

    # 建立一個 2x2 的圖表方格
    plt.figure(figsize=(12, 10))
    plt.suptitle('FaceLock Loss Convergence', fontsize=16)

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
    plt.title('Face Recognition Score (loss_cvl)')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)

    # 圖 3: Encoder Loss (潛在空間距離)
    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (loss_encoder)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 圖 4: LPIPS Loss (影像感知相似度)
    plt.subplot(2, 2, 4)
    plt.plot(history['loss_lpips'], color='purple')
    plt.title('Perceptual Similarity (loss_lpips)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 調整排版並儲存圖表
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "facelock_loss_convergence.png"
    plt.savefig(save_path) 
    print(f"Loss plot saved to: {save_path}")
    # plt.show() # 在 Agg 模式下無法運作
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

def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, plot_history=False): 
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max).half()
    pbar = tqdm(range(iters))
    
    vae = model.vae
    X_adv.requires_grad_(True)
    clean_latent = vae.encode(X).latent_dist.mean

    history = {
        'total_loss': [],
        'loss_cvl': [],
        'loss_encoder': [],
        'loss_lpips': []
    }

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)

        loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        loss_lpips = lpips_fn(image, X)
        loss = -loss_cvl * (5 if i >= iters * 0.35 else 0.0) + loss_encoder *0.5 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(loss_cvl=loss_cvl.item(), loss_encoder=loss_encoder.item(), loss_lpips=loss_lpips.item(), loss=loss.item())

        history['total_loss'].append(loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())
        history['loss_lpips'].append(loss_lpips.item())

    if plot_history:
        plot_facelock_history(history)

    return X_adv, history

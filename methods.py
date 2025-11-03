import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from utils import compute_score
import pdb

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
    
    # [CPU 支援] 移除 .cuda(), 使用 .to(X.device)
    w = torch.zeros_like(X, requires_grad=True).to(X.device) 
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        # pdb.set_trace()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    encoder = model.vae.encode
    
    # [CPU 支援] 移除 .half().cuda()
    # 使用 X.device 和 X.dtype 來創建隨機噪聲，確保匹配
    rand_noise = (torch.rand(*X.shape, device=X.device, dtype=X.dtype) * 2 * eps - eps)
    X_adv = torch.clamp(X.clone().detach() + rand_noise, min=clamp_min, max=clamp_max)

    if not targeted:
        loss_fn = nn.MSELoss()
        with torch.no_grad(): # 確保 clean_latent 計算不佔用梯度
            clean_latent = encoder(X).latent_dist.mean
            
    pbar = tqdm(range(iters))
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

    return X_adv

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    vae = model.vae
    
    # [CPU 支援] 移除 .half().cuda()
    # 使用 X.device 和 X.dtype 來創建隨機噪聲，確保匹配
    rand_noise = (torch.rand(*X.shape, device=X.device, dtype=X.dtype) * 2 * eps - eps)
    X_adv = torch.clamp(X.clone().detach() + rand_noise, min=clamp_min, max=clamp_max)

    pbar = tqdm(range(iters))
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

    return X_adv

def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    # [CPU 支援] 使用 X.device 和 X.dtype 確保匹配
    rand_noise = (torch.rand(*X.shape, device=X.device, dtype=X.dtype) * 2 * eps - eps)
    X_adv = torch.clamp(X.clone().detach() + rand_noise, min=clamp_min, max=clamp_max)
    
    pbar = tqdm(range(iters))
    
    vae = model.vae
    X_adv.requires_grad_(True)
    with torch.no_grad(): # 確保 clean_latent 計算不佔用梯度
        clean_latent = vae.encode(X).latent_dist.mean

# --- 權重設定 (方案 C：高權重平衡) ---
    # 我們的目標：臉部和背景 "都" 要有強烈的擾動
    lambda_cvl = 1.0     # (高) 保留臉部防禦的原始強度
    lambda_encoder = 3.0 # (極高) 大幅提高全圖的潛在空間防禦
    lambda_lpips = 3.0   # (極高) 大幅提高全圖的感知防禦
    # --- 修改結束 ---

    for i in pbar:
        # actual_step_size = step_size
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)

        # 確保 LPIPS 和 compute_score 的輸入是 .float() (float32)
        loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        loss_lpips = lpips_fn(image.float(), X.float())
        
        # --- 「方法一」的損失計算 ---
        # 2. (註解掉) 這是原始的 loss 計算
        # loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + loss_encoder * 0.2 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
        
        # 3. (新的) 使用 lambda 權重重新組合 loss，並移除 if 條件
        loss = -loss_cvl * lambda_cvl + loss_encoder * lambda_encoder + loss_lpips * lambda_lpips
        # --- 修改結束 ---
        
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(loss_cvl=loss_cvl.item(), loss_encoder=loss_encoder.item(), loss_lpips=loss_lpips.item(), loss=loss.item())

    return X_adv

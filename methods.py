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

# --- 正在載入已修正的 METHODS.PY v4 (已修復 IndexError) ---
# (您可以自行修改這個標記)

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
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
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
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    return X_adv

# --- 這是「方向一」優化後的 FACELOCK ---
def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    
    # --- 1. PREPARATION (NEW) ---
    # 從管線中獲取所有組件
    vae = model.vae
    unet = model.unet
    scheduler = model.scheduler
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    device = X.device

    # 新的模擬超參數
    # SIMULATION_TIMESTEP = 200 # (已移除 - 這是錯誤的來源)
    SIMULATION_STEPS = 5      # 淨化步數 (S=5)

    # 預先計算「無條件嵌入」（空指令）
    with torch.no_grad():
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(X.dtype)

    # --- *** 這是錯誤修復 v2 *** ---
    # 1. 提前初始化排程器
    scheduler.set_timesteps(num_inference_steps=50, device=device)
    
    # 2. 不再使用任意的 '200'，而是從排程器中 *合法地* 選取一個起始點
    #    我們從 50 步的中間點 (索引 25) 開始
    t_start_index = len(scheduler.timesteps) // 2
    t_start = scheduler.timesteps[t_start_index] # 獲取 *合法的* 時間步 (例如 481)
    
    # 3. 獲取我們要運行的 S 步
    timesteps_to_run = scheduler.timesteps[t_start_index : t_start_index + SIMULATION_STEPS]
    # --- *** 修復結束 v2 *** ---

    # --- (END NEW PREPARATION) ---

    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(device), min=clamp_min, max=clamp_max).half()
    pbar = tqdm(range(iters))
    
    X_adv.requires_grad_(True)
    clean_latent = vae.encode(X).latent_dist.mean # [1, 4, 64, 64]

    # --- 使用我們測試出的最佳參數 (Test 6) ---
    lambda_encoder = 0.0 
    lambda_cvl = 2.0   
    lambda_lpips = 2.0 
    # ---

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        # --- 2. REPLACING STEP 1 (VAE SIMULATION) ---
        # 這是「微型擴散模擬」
        
        # a. 編碼 (Algorithm 1, line 4)
        latent = vae.encode(X_adv).latent_dist.mean # [1, 4, 64, 64]

        # b. 添加噪聲 (模擬淨化的起始點)
        noise = torch.randn_like(latent)
        # 這一行現在可以安全執行了，因為 t_start (例如 481) 100% 在 scheduler.timesteps 中
        noisy_latent = scheduler.add_noise(latent, noise, t_start) # t_start 是一個 tensor
        
        # c. (已移至迴圈外)

        # d. 執行 S 步去噪迴圈 (模擬淨化)
        simulated_latent = noisy_latent
        
        # *** 這是 InstructPix2Pix 的關鍵 ***
        # 我們需要 `clean_latent` 作為 U-Net 的額外條件 (4 個通道)
        image_latent_cond = clean_latent # [1, 4, 64, 64]

        with torch.no_grad(): # 我們不需要在模擬內部計算梯度
            for t in timesteps_to_run:
                
                # --- *** 這是錯誤修復 *** ---
                # 串聯 `simulated_latent` (4 通道) 和 `image_latent_cond` (4 通道)
                # 得到 U-Net 期望的 8 通道輸入
                # [1, 4, 64, 64] + [1, 4, 64, 64] -> [1, 8, 64, 64]
                latent_model_input = torch.cat([simulated_latent, image_latent_cond], dim=1)
                # --- *** 修復結束 *** ---

                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # 預測噪聲 (使用空指令)
                noise_pred = unet(
                    latent_model_input, # 現在這是 8 通道, shape 正確
                    t,
                    encoder_hidden_states=uncond_embeddings
                ).sample

                # 計算上一步 (去噪)
                simulated_latent = scheduler.step(noise_pred, t, simulated_latent).prev_sample

        # e. 解碼 (Algorithm 1, line 5 - 但使用 *simulated_latent*)
        image = vae.decode(simulated_latent).sample.clip(-1, 1)
        # --- (END OF REPLACEMENT) ---

        # --- 3. COMPUTE LOSSES (在新的 'image' 上計算) ---
        loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent) # 'latent' 是未加噪聲的
        loss_lpips = lpips_fn(image, X)
        
        # 損失函式 (Test 6: 移除預熱, -lpips)
        loss = -loss_cvl * lambda_cvl + \
               loss_encoder * lambda_encoder - \
               loss_lpips * lambda_lpips
        # --- (END OF LOSS COMPUTATION) ---

        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(loss_cvl=loss_cvl.item(), loss_encoder=loss_encoder.item(), loss_lpips=loss_lpips.item(), loss=loss.item())

    return X_adv

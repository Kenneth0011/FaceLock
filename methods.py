import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import cv2
import dlib
from tqdm import tqdm
from utils import compute_score
import pdb

# =========================================================
# [修正錯誤] 強制設定 Matplotlib 後端 (必須在 import matplotlib 之前)
# =========================================================
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 輔助功能：遮罩生成、繪圖、檢查
# ==============================================================================

_dlib_detector = None

def get_face_mask(image_tensor, pad=20, blur_sigma=15):
    """
    自動偵測人臉並生成遮罩
    """
    global _dlib_detector
    if _dlib_detector is None:
        _dlib_detector = dlib.get_frontal_face_detector()

    img_np = image_tensor.detach().cpu().squeeze()
    if img_np.dim() == 3:
        img_np = img_np.permute(1, 2, 0).numpy()
    
    # 轉為 0-255 uint8
    if img_np.min() < 0:
        img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
    elif img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    # 處理灰階圖
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    H, W = img_np.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    
    # Dlib 偵測
    dets = _dlib_detector(img_np, 1)
    
    if len(dets) > 0:
        for d in dets:
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            y1 = max(0, y1 - int(pad * 1.5))
            y2 = min(H, y2 + pad)
            x1 = max(0, x1 - pad)
            x2 = min(W, x2 + pad)
            mask[y1:y2, x1:x2] = 1.0
    else:
        # 備案：中心裁切
        print("[Warning] No face detected. Using center crop mask.")
        cy, cx = H // 2, W // 2
        h_r, w_r = H // 3, W // 3
        mask[cy-h_r:cy+h_r, cx-w_r:cx+w_r] = 1.0

    # 羽化邊緣
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    
    mask_tensor = torch.from_numpy(mask).to(image_tensor.device)
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
        
    mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
    return mask_tensor

def save_mask_check(image_tensor, mask_tensor, save_path="facelock_mask_check.png"):
    """
    儲存遮罩檢查圖
    """
    print(f"Saving mask visualization to {save_path}...")
    
    img = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    mask = mask_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    mask_display = mask[:, :, 0] 
    masked_img = img * mask

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(mask_display, cmap='gray'); plt.title("Generated Mask"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(masked_img); plt.title("Attack Area Overlay"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_history(history, save_path="facelock_loss_convergence.png"):
    """
    繪製 Loss 曲線圖
    """
    print(f"Plotting losses to {save_path}...")
    plt.figure(figsize=(12, 10))
    plt.suptitle('FaceLock Loss Convergence', fontsize=16)

    plt.subplot(2, 2, 1); plt.plot(history['total_loss']); plt.title('Total Loss'); plt.grid(True)
    plt.subplot(2, 2, 2); plt.plot(history['loss_cvl'], color='red'); plt.title('Face Recognition Score'); plt.grid(True)
    plt.subplot(2, 2, 3); plt.plot(history['loss_encoder'], color='green'); plt.title('Encoder MSE Loss'); plt.grid(True)
    plt.subplot(2, 2, 4); plt.plot(history['loss_lpips'], color='purple'); plt.title('Perceptual Similarity'); plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 2. FaceLock 主程式 (優化版)
# ==============================================================================

def facelock(X, model, aligner, fr_model, lpips_fn, 
             eps=0.03, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot=True):
    
    # 1. 取得遮罩
    mask = get_face_mask(X, pad=30, blur_sigma=15)
    
    # 輸出遮罩檢查圖
    if plot:
        save_mask_check(X, mask, save_path="facelock_mask_check.png")

    # 初始化
    noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
    X_adv = torch.clamp(X.clone().detach() + noise * mask, min=clamp_min, max=clamp_max).half()
    
    history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': [], 'loss_lpips': []}
    
    pbar = tqdm(range(iters), desc="FaceLock (Masked)")
    
    vae = model.vae
    X_adv.requires_grad_(True)
    clean_latent = vae.encode(X).latent_dist.mean.detach()

    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)

        loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        loss_lpips = lpips_fn(image, X)
        
        loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + \
               loss_encoder * 0.2 + \
               loss_lpips * (1 if i > iters * 0.25 else 0.0)
               
        grad, = torch.autograd.grad(loss, [X_adv])
        grad = grad * mask # 梯度遮罩
        
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        delta = torch.clamp(X_adv - X, min=-eps, max=eps)
        X_adv = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
        X_adv.data = X_adv.data * mask + X.data * (1 - mask) # 背景強制還原
        X_adv.grad = None

        history['total_loss'].append(loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())
        history['loss_lpips'].append(loss_lpips.item())

        pbar.set_postfix(
            cvl=f"{loss_cvl.item():.3f}", 
            lpips=f"{loss_lpips.item():.3f}", 
            loss=f"{loss.item():.3f}"
        )
    
    if plot:
        plot_loss_history(history)

    return X_adv

# ==============================================================================
# 3. 傳統攻擊函數 (Legacy Methods) - 必須保留以防報錯
# ==============================================================================

def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean.detach()

    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted:
            return latents.norm()
        else:
            return -torch.norm(latents - clean_latents, p=2, dim=-1)
    
    w = torch.zeros_like(X, requires_grad=True).cuda()
    pbar = tqdm(range(iters), desc="CW Attack")
    optimizer = optim.Adam([w], lr=lr)

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        pbar.set_postfix(loss=cost.item())
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean.detach()
        
    pbar = tqdm(range(iters), desc="Encoder Attack")
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

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        pbar.set_postfix(loss=loss.item())

    return X_adv

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters), desc="VAE Attack")
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv.requires_grad_()
        image = vae(X_adv).sample
        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        pbar.set_postfix(loss=loss.item())

    return X_adv

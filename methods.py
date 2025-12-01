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

# [新增] 引入 matplotlib 用於繪圖
import matplotlib
# 強制使用 Agg 後端，防止在沒有螢幕的伺服器上報錯
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pdb

# ==============================================================================
# 1. 輔助功能：自動生成人臉遮罩 & 繪圖
# ==============================================================================

_dlib_detector = None

def get_face_mask(image_tensor, pad=20, blur_sigma=15):
    """
    自動偵測人臉並生成遮罩 (與上一版相同)
    """
    global _dlib_detector
    if _dlib_detector is None:
        _dlib_detector = dlib.get_frontal_face_detector()

    img_np = image_tensor.detach().cpu().squeeze()
    if img_np.dim() == 3:
        img_np = img_np.permute(1, 2, 0).numpy()
    
    if img_np.min() < 0:
        img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
    elif img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    H, W = img_np.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    
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
        print("[Warning] No face detected. Using center crop mask.")
        cy, cx = H // 2, W // 2
        h_r, w_r = H // 3, W // 3
        mask[cy-h_r:cy+h_r, cx-w_r:cx+w_r] = 1.0

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
    
    mask_tensor = torch.from_numpy(mask).to(image_tensor.device)
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
        
    mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
    return mask_tensor

def plot_loss_history(history, save_path="facelock_loss_convergence.png"):
    """
    [新增] 繪製 Loss 曲線圖
    """
    print(f"Plotting losses to {save_path}...")
    plt.figure(figsize=(12, 10))
    plt.suptitle('FaceLock Loss Convergence', fontsize=16)

    # 1. Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 2. Face Recognition Score
    plt.subplot(2, 2, 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (loss_cvl)')
    plt.xlabel('Iteration')
    plt.ylabel('Score (Lower is better for attack)')
    plt.grid(True)

    # 3. Encoder MSE
    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (loss_encoder)')
    plt.xlabel('Iteration')
    plt.grid(True)

    # 4. LPIPS
    plt.subplot(2, 2, 4)
    plt.plot(history['loss_lpips'], color='purple')
    plt.title('Perceptual Similarity (loss_lpips)')
    plt.xlabel('Iteration')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print("Plot saved.")

# ==============================================================================
# 2. FaceLock 主程式 (整合記錄功能)
# ==============================================================================

def facelock(X, model, aligner, fr_model, lpips_fn, 
             eps=0.03, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot=True): # [新增] plot 參數控制是否繪圖
    
    # 1. 取得遮罩
    mask = get_face_mask(X, pad=30, blur_sigma=15)
    
    # 初始化
    noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
    X_adv = torch.clamp(X.clone().detach() + noise * mask, min=clamp_min, max=clamp_max).half()
    
    # [新增] 初始化 History 字典
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
        grad = grad * mask
        
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        delta = torch.clamp(X_adv - X, min=-eps, max=eps)
        X_adv = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
        X_adv.data = X_adv.data * mask + X.data * (1 - mask)
        X_adv.grad = None

        # [新增] 記錄數據
        history['total_loss'].append(loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())
        history['loss_lpips'].append(loss_lpips.item())

        pbar.set_postfix(
            cvl=f"{loss_cvl.item():.3f}", 
            lpips=f"{loss_lpips.item():.3f}", 
            loss=f"{loss.item():.3f}"
        )
    
    # [新增] 攻擊結束後繪圖
    if plot:
        plot_loss_history(history)

    return X_adv

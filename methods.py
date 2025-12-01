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

import matplotlib
matplotlib.use('Agg') # 防止伺服器端報錯
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

def save_mask_check(image_tensor, mask_tensor, save_path="facelock_mask_check.png"):
    """
    [新增] 儲存遮罩檢查圖
    將原圖、遮罩、與遮罩後的圖畫在一起，方便檢查
    """
    print(f"Saving mask visualization to {save_path}...")
    
    # 處理原圖 (Tensor -> Numpy, -1~1 -> 0~1)
    img = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    # 處理遮罩 (Tensor -> Numpy)
    mask = mask_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    # 遮罩通常是3通道一樣的，取第一個通道顯示即可
    mask_display = mask[:, :, 0] 

    # 處理疊加圖 (顯示攻擊區域)
    masked_img = img * mask

    plt.figure(figsize=(15, 5))
    
    # 1. 原圖
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # 2. 遮罩 (黑白)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title("Generated Mask (White=Attack)")
    plt.axis('off')

    # 3. 攻擊區域預覽
    plt.subplot(1, 3, 3)
    plt.imshow(masked_img)
    plt.title("Attack Area Overlay")
    plt.axis('off')

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

    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (loss_cvl)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('Encoder MSE Loss (loss_encoder)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['loss_lpips'], color='purple')
    plt.title('Perceptual Similarity (loss_lpips)')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 2. FaceLock 主程式
# ==============================================================================

def facelock(X, model, aligner, fr_model, lpips_fn, 
             eps=0.03, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot=True): # plot=True 會同時輸出 Loss圖 和 Mask檢查圖
    
    # 1. 取得遮罩
    mask = get_face_mask(X, pad=30, blur_sigma=15)
    
    # [新增] 如果開啟繪圖，這裡直接輸出遮罩檢查圖
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
        grad = grad * mask
        
        X_adv = X_adv + grad.detach().sign() * actual_step_size

        delta = torch.clamp(X_adv - X, min=-eps, max=eps)
        X_adv = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
        X_adv.data = X_adv.data * mask + X.data * (1 - mask)
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

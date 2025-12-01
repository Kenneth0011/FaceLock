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
# 強制設定 Matplotlib 後端
# =========================================================
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 輔助功能：遮罩生成、邊緣提取、繪圖
# ==============================================================================

_dlib_detector = None

def get_face_mask(image_tensor, pad=20, blur_sigma=15):
    """
    自動偵測人臉並生成遮罩
    """
    global _dlib_detector
    if _dlib_detector is None:
        _dlib_detector = dlib.get_frontal_face_detector()

    img_np = image_tensor.detach().cpu().squeeze().float()
    if img_np.dim() == 3:
        img_np = img_np.permute(1, 2, 0).numpy()
    else:
        img_np = img_np.numpy()
    
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

def get_edge_mask(mask_tensor, thickness=15):
    """
    [新增] 從 Face Mask 提取邊緣區域 (甜甜圈形狀)
    """
    mask_np = mask_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    mask_np = mask_np[:, :, 0] # 取單通道
    
    kernel = np.ones((thickness, thickness), np.uint8)
    
    # 膨脹 (變胖)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    # 腐蝕 (變瘦)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    
    # 相減得到邊緣
    edge = dilated - eroded
    
    # 稍微模糊邊緣
    edge = cv2.GaussianBlur(edge, (5, 5), 0)
    
    edge_tensor = torch.from_numpy(edge).to(mask_tensor.device)
    edge_tensor = edge_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return edge_tensor

def save_mask_check(image_tensor, mask_tensor, edge_mask_tensor, save_path="facelock_mask_check.png"):
    """
    [更新] 顯示原圖、臉部遮罩、以及攻擊重點(邊緣)
    """
    print(f"Saving mask visualization to {save_path}...")
    
    img = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).float().numpy()
    if img.min() < 0: img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    mask = mask_tensor.detach().cpu().squeeze().permute(1, 2, 0).float().numpy()[:, :, 0]
    edge = edge_mask_tensor.detach().cpu().squeeze().permute(1, 2, 0).float().numpy()[:, :, 0]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image", y=1.02)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Face Area (Texture Attack)", y=1.02)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(edge, cmap='magma')
    plt.title("Boundary Area (Blur Attack)", y=1.02)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 修正標題被切掉的問題
    plt.savefig(save_path)
    plt.close()

def plot_loss_history(history, save_path="facelock_loss_convergence.png"):
    """
    [更新] 繪製新的 Loss 曲線 (Boundary & Texture)
    """
    print(f"Plotting losses to {save_path}...")
    plt.figure(figsize=(12, 10))
    plt.suptitle('Anti-Inpainting Loss Convergence', fontsize=16)

    plt.subplot(2, 2, 1); plt.plot(history['total_loss']); plt.title('Total Loss'); plt.grid(True)
    plt.subplot(2, 2, 2); plt.plot(history['loss_boundary'], color='orange'); plt.title('Boundary Disruption (Lower is Better)'); plt.grid(True)
    plt.subplot(2, 2, 3); plt.plot(history['loss_texture'], color='green'); plt.title('Texture Disruption (Lower is Better)'); plt.grid(True)
    plt.subplot(2, 2, 4); plt.plot(history['loss_lpips'], color='purple'); plt.title('Visual Similarity (LPIPS)'); plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 2. 抗編輯攻擊主程式 (取代原本的 Facelock 邏輯)
# ==============================================================================

def facelock(X, model, aligner, fr_model, lpips_fn, 
             eps=0.03, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot=True):
    """
    Anti-Inpainting / Boundary Attack
    目標：破壞 VAE 在臉部邊緣的特徵，使編輯模型無法正確分割前景背景。
    """
    
    # 1. 取得遮罩
    # face_mask: 臉部區域 (我們要在這裡做輕微的紋理破壞)
    face_mask = get_face_mask(X, pad=20, blur_sigma=10)
    
    # edge_mask: 邊緣區域 (我們要在這裡做重度的邊界混淆)
    edge_mask = get_edge_mask(face_mask, thickness=20)
    
    # 組合攻擊遮罩 (視覺化用)
    if plot:
        save_mask_check(X, face_mask, edge_mask, save_path="facelock_mask_check.png")

    # 初始化擾動 (全圖加噪，允許自然過渡)
    noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
    X_adv = torch.clamp(X.clone().detach() + noise, min=clamp_min, max=clamp_max).half()
    
    # 定義攻擊目標
    # 1. 邊界目標：我們希望邊界看起來像「雜訊」或「背景」，而不是清晰的線條
    # 使用隨機高斯雜訊作為邊界的目標
    vae = model.vae
    with torch.no_grad():
        clean_latent = vae.encode(X).latent_dist.mean.detach()
        target_boundary_latent = torch.randn_like(clean_latent) * 1.5 # 強度較高的雜訊
        
        # 2. 紋理目標：臉部內部稍微偏離原圖，但不至於毀容
        # 使用稍微偏移的 latent 作為目標
        target_texture_latent = clean_latent + torch.randn_like(clean_latent) * 0.5

    history = {'total_loss': [], 'loss_boundary': [], 'loss_texture': [], 'loss_lpips': []}
    
    print(f"Starting Anti-Inpainting Attack (iters={iters}, eps={eps})...")
    pbar = tqdm(range(iters), desc="Optimizing")
    
    X_adv.requires_grad_(True)
    
    # 使用 Adam 優化器通常比 PGD 收斂更好 (針對這種 MSE Loss)
    # 如果效果不好，可以換回 PGD
    optimizer = optim.Adam([X_adv], lr=step_size)

    for i in pbar:
        # VAE Encode -> Decode
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)

        # --- Loss 計算 ---
        
        # 1. Boundary Loss (邊界混淆)
        # 強迫邊緣區域的 Latent 接近隨機雜訊 --> 破壞語義分割
        loss_boundary = F.mse_loss(latent * edge_mask, target_boundary_latent * edge_mask)
        
        # 2. Texture Loss (紋理破壞)
        # 強迫臉部內部的 Latent 發生偏移 --> 防止特徵被完美識別
        loss_texture = F.mse_loss(latent * face_mask, target_texture_latent * face_mask)
        
        # 3. LPIPS (視覺維持)
        # 確保攻擊後的圖看起來還是正常的
        loss_lpips = lpips_fn(image, X)
        
        # 權重配置
        w_bound = 20.0  # 最重要：邊界必須爛掉
        w_tex = 2.0     # 次要：臉部稍微變異
        w_lpips = 5.0   # 約束：肉眼看不出來
        
        # 我們希望 Minimize MSE (接近目標雜訊) 和 Minimize LPIPS (接近原圖)
        loss = loss_boundary * w_bound + loss_texture * w_tex + loss_lpips * w_lpips
        
        optimizer.zero_grad()
        loss.backward()
        
        # 如果使用 Adam，直接 step
        # 如果要嚴格遵守 eps 限制，可以在 step 後做 projection
        optimizer.step()
        
        # Constraints (Project & Clip)
        with torch.no_grad():
            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
            X_adv.data = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
            # [關鍵] 不再強制還原背景，允許攻擊雜訊擴散
            # 但我們可以選擇「遠離臉部的背景」還原，保留「臉部周圍」的攻擊
            # 這裡簡單起見，全圖允許微幅攻擊，這樣效果最好

        # 記錄
        history['total_loss'].append(loss.item())
        history['loss_boundary'].append(loss_boundary.item())
        history['loss_texture'].append(loss_texture.item())
        history['loss_lpips'].append(loss_lpips.item())

        pbar.set_postfix(
            bound=f"{loss_boundary.item():.4f}", 
            tex=f"{loss_texture.item():.4f}", 
            lpips=f"{loss_lpips.item():.3f}"
        )
    
    if plot:
        plot_loss_history(history)

    return X_adv.detach()

# ==============================================================================
# 3. 傳統攻擊函數 (Legacy Methods) - 保留以防報錯
# ==============================================================================

def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean.detach()
    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted: return latents.norm()
        else: return -torch.norm(latents - clean_latents, p=2, dim=-1)
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

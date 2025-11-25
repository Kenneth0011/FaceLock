import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import dlib
import os
from tqdm import tqdm

# [重要] 強制將 Matplotlib 後端設為 Agg，避免在 Kaggle/無螢幕環境下報錯
# 這行必須在 import matplotlib 之前執行
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt

# 假設 utils.py 已經存在並包含 compute_score
from utils import compute_score

# ==============================================================================
# 第一部分：傳統攻擊方法 (Legacy Methods)
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
    optimizer = optim.Adam([w], lr=lr)
    
    history = {'total_loss': [], 'loss1': [], 'loss2': []} 
    pbar = tqdm(range(iters), desc="CW L2 Attack")

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        history['total_loss'].append(cost.item())
        history['loss1'].append(loss1.item())
        history['loss2'].append(loss2.item())
        
        pbar.set_postfix(cost=cost.item())
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv, history 

def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda(), min=clamp_min, max=clamp_max).half()
    
    clean_latent = encoder(X).latent_dist.mean.detach()
    history = {'loss': []} 
    
    pbar = tqdm(range(iters), desc="Encoder Attack")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        
        if targeted:
            loss = latent.norm() # Target towards 0
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = nn.MSELoss()(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size # Ascent to maximize error

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        history['loss'].append(loss.item())
        pbar.set_postfix(loss=loss.item())

    return X_adv, history 

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda(), min=clamp_min, max=clamp_max).half()
    
    history = {'loss': []} 
    pbar = tqdm(range(iters), desc="VAE Attack")
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample
        
        # Target towards 0 tensor (gray image)
        loss = image.norm() 
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        
        history['loss'].append(loss.item())
        pbar.set_postfix(loss=loss.item())

    return X_adv, history 


# ==============================================================================
# 第二部分：FaceLock 攻擊器類別 (FaceLockAttacker)
# 包含：
# 1. 擴大的遮罩邏輯 (額頭延伸 + Dilation)
# 2. Gradient Descent 優化
# 3. 梯度 Masking 保護背景
# ==============================================================================

class FaceLockAttacker:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        print("[FaceLock] Initializing Dlib detector...")
        self.detector = dlib.get_frontal_face_detector()
        self.has_landmarks = False
        
        if os.path.exists(predictor_path):
            try:
                self.predictor = dlib.shape_predictor(predictor_path)
                self.has_landmarks = True
                print(f"[FaceLock] Loaded landmark predictor: {predictor_path}")
            except Exception as e:
                print(f"[FaceLock] Error loading predictor: {e}")
        else:
            print(f"[FaceLock] Warning: {predictor_path} not found. Will use bounding box mask (less accurate).")

    def get_face_mask(self, image_tensor, pad=50, blur_sigma=20):
        """
        生成擴大的臉部遮罩。
        pad: 擴張像素量 (預設 50，覆蓋額頭與邊緣)
        blur_sigma: 邊緣模糊程度 (預設 20，平滑過渡)
        """
        # 轉為 Numpy image (H, W, 3), 範圍 [0, 255]
        img_np = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        
        # 處理標準化
        if img_np.min() < 0:
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
        elif img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)
        
        dets = self.detector(img_np, 1)
        
        if len(dets) == 0:
            print("[FaceLock] Warning: No face detected! Mask will be empty.")
            return torch.ones_like(image_tensor) # 無法偵測時，預設攻擊全圖或不攻擊(視需求)

        d = dets[0]

        if self.has_landmarks:
            shape = self.predictor(img_np, d)
            points = []
            for i in range(68):
                points.append((shape.part(i).x, shape.part(i).y))
            
            # --- 手動向上延伸以覆蓋額頭 (Forehead Extension) ---
            # 利用眉毛和鼻子的距離來估算額頭高度
            eyebrow_y_min = min(shape.part(19).y, shape.part(24).y) # 眉毛最高點
            nose_y = shape.part(27).y # 鼻樑點
            
            # 估算額頭高度 (通常是眉毛到鼻子距離的 1.5 倍左右)
            forehead_height = int(abs(nose_y - eyebrow_y_min) * 1.5)
            
            # 取得臉部最左和最右的 X 座標 (大約是臉頰寬度)
            face_left_x = shape.part(0).x
            face_right_x = shape.part(16).x
            
            # 加入額頭頂部的兩個點 (左上, 右上)
            points.append((face_left_x, eyebrow_y_min - forehead_height))
            points.append((face_right_x, eyebrow_y_min - forehead_height))
            # ------------------------------------------------
            
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            # 備用方案：方框，並手動往上擴大
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            # 往上延伸 pad 距離
            y1_extended = max(0, y1 - pad)
            mask[y1_extended:y2, x1:x2] = 1.0

        # --- 擴大遮罩 (Dilation) ---
        kernel = np.ones((pad, pad), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # --- 邊緣模糊 (Blur) ---
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
        
        # 轉回 Tensor
        mask_tensor = torch.from_numpy(mask).view(1, 1, H, W).repeat(1, 3, 1, 1)
        mask_tensor = mask_tensor.to(image_tensor.device)
        
        return mask_tensor

    def attack(self, X, model, aligner, fr_model, lpips_fn, 
               eps=0.03, step_size=0.01, iters=100, 
               clamp_min=-1, clamp_max=1, plot_history=False):
        
        # 1. Mask 保持剛才的設定 (覆蓋額頭)
        mask = self.get_face_mask(X, pad=50, blur_sigma=20)
        
        noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
        X_adv = X.clone().detach() + noise * mask
        X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max).half()

        vae = model.vae
        clean_latent = vae.encode(X).latent_dist.mean.detach()
        momentum = torch.zeros_like(X_adv).detach()
        
        history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': [], 'loss_lpips': []}
        
        print(f"[FaceLock] Starting HARD MODE attack ({iters} iters)...")
        pbar = tqdm(range(iters))

        for i in pbar:
            actual_step_size = step_size - (step_size - step_size / 100) / iters * i
            
            X_adv.requires_grad_(True)
            
            latent = vae.encode(X_adv).latent_dist.mean
            image = vae.decode(latent).sample.clip(clamp_min, clamp_max)

            # Loss 計算
            loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image, X)

            # --- 關鍵修改：權重策略調整 ---
            # 舊設定: w_cvl = 2.0, wait 35%
            # 新設定: w_cvl = 10.0 (加強 5 倍), wait 0% (全程攻擊)
            
            w_cvl = 10.0  # 強制壓低 FR 分數
            w_lpips = 2.0 # 稍微提高 LPIPS 權重以平衡視覺
            w_enc = 0.1   # 降低結構束縛，讓特徵可以變形
            
            # 如果你想要保留一點點熱身，可以用下面這行，但我建議直接全開
            # w_cvl = 10.0 if i >= iters * 0.1 else 0.0 

            loss = loss_cvl * w_cvl + loss_encoder * w_enc + loss_lpips * w_lpips
            # ---------------------------
            
            grad, = torch.autograd.grad(loss, [X_adv])
            grad = grad * mask 

            grad_norm = torch.norm(grad, p=1)
            grad = grad / (grad_norm + 1e-10)
            momentum = momentum + grad
            
            X_adv = X_adv - momentum.sign() * actual_step_size

            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
            X_adv = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
            
            X_adv.data = X_adv.data * mask + X.data * (1 - mask)
            X_adv = X_adv.detach()

            pbar.set_postfix(cvl=loss_cvl.item(), total=loss.item())
            history['total_loss'].append(loss.item())
            history['loss_cvl'].append(loss_cvl.item())
            history['loss_encoder'].append(loss_encoder.item())
            history['loss_lpips'].append(loss_lpips.item())

        if plot_history:
            self._plot_history(history)

        return X_adv, history

    def _plot_history(self, history):
        print("Plotting losses...")
        plt.figure(figsize=(12, 10))
        plt.suptitle('FaceLock Loss Convergence (Optimized)', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.plot(history['total_loss'])
        plt.title('Total Loss')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(history['loss_cvl'], color='red')
        plt.title('Face Recog Score (Minimize)')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(history['loss_encoder'], color='green')
        plt.title('Encoder MSE (Minimize)')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(history['loss_lpips'], color='purple')
        plt.title('LPIPS Loss (Minimize)')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = "facelock_loss_convergence.png"
        plt.savefig(save_path)
        print(f"Loss plot saved to: {save_path}")
        plt.close()

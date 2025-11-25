import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import dlib
import os
from tqdm import tqdm
import matplotlib
# 強制設定 matplotlib 後端，避免在無螢幕環境 (如 Kaggle) 報錯
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt

# 假設 utils.py 裡有 compute_score，請確保該檔案存在
from utils import compute_score

# ==============================================================================
# 第一部分：傳統攻擊方法 (Legacy Methods)
# 包含 CW, Encoder Attack, VAE Attack
# 已更新回傳格式為 (X_adv, history)
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
# 包含 Mask 生成、優化後的 Facelock 演算法、以及繪圖功能
# ==============================================================================

class FaceLockAttacker:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        初始化：載入 Dlib 模型
        """
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

    def get_face_mask(self, image_tensor, pad=15, blur_sigma=15):
        """
        生成臉部遮罩 (Soft Mask)
        image_tensor: (1, 3, H, W)
        """
        # 轉為 Numpy image (H, W, 3), 範圍 [0, 255]
        img_np = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        
        # 處理標準化: 假設輸入可能是 [-1, 1] 或 [0, 1]
        if img_np.min() < 0:
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
        elif img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)
        
        # 偵測人臉
        dets = self.detector(img_np, 1)
        
        if len(dets) == 0:
            print("[FaceLock] Warning: No face detected! Mask will be empty (no attack applied).")
            # 回傳全黑 (不攻擊) 或全白 (全攻擊)，這裡設為全白以防萬一，但最好是回傳警告
            return torch.ones_like(image_tensor)

        d = dets[0] # 取第一張臉

        if self.has_landmarks:
            # 使用 68 特徵點
            shape = self.predictor(img_np, d)
            points = []
            for i in range(68):
                points.append((shape.part(i).x, shape.part(i).y))
            
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            # 備用方案：方框
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            mask[y1:y2, x1:x2] = 1.0

        # 處理遮罩邊緣 (Dilation + Gaussian Blur)
        kernel = np.ones((pad, pad), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)
        
        # 轉回 Tensor 並送到 GPU
        mask_tensor = torch.from_numpy(mask).view(1, 1, H, W).repeat(1, 3, 1, 1)
        mask_tensor = mask_tensor.to(image_tensor.device)
        
        return mask_tensor

    def attack(self, X, model, aligner, fr_model, lpips_fn, 
               eps=0.03, step_size=0.01, iters=100, 
               clamp_min=-1, clamp_max=1, plot_history=False):
        """
        執行 Facelock 攻擊 (Optimized + Masked)
        """
        # 1. 自動生成 Mask
        mask = self.get_face_mask(X)
        
        # 2. 初始化攻擊圖像 (只在 Mask 區域加噪聲)
        noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
        # 初始只在臉上加噪聲
        X_adv = X.clone().detach() + noise * mask
        X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max).half()

        vae = model.vae
        clean_latent = vae.encode(X).latent_dist.mean.detach()
        momentum = torch.zeros_like(X_adv).detach()
        
        history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': [], 'loss_lpips': []}
        
        print(f"[FaceLock] Starting attack ({iters} iters)...")
        pbar = tqdm(range(iters))

        for i in pbar:
            actual_step_size = step_size - (step_size - step_size / 100) / iters * i
            
            X_adv.requires_grad_(True)
            
            # --- Forward Pass ---
            latent = vae.encode(X_adv).latent_dist.mean
            image = vae.decode(latent).sample.clip(clamp_min, clamp_max)

            # --- Loss 計算 (Minimize All Strategy) ---
            # loss_cvl: 我們希望 Similarity 越低越好。
            # 但 PGD 通常是 Minimize Loss。
            # 如果 compute_score 回傳的是 Cosine Similarity (數值越大越像)，
            # 我們應該 Minimize (Similarity)。
            # 注意：請確認 utils.py 裡的 compute_score 是否回傳 Cosine Similarity。
            loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
            
            # loss_encoder & lpips: 我們希望越像越好 -> Minimize MSE
            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image, X)

            # 權重排程 (Scheduler)
            w_cvl = 2.0 if i >= iters * 0.35 else 0.0
            w_lpips = 1.0 if i > iters * 0.25 else 0.0
            w_enc = 0.2

            # 總 Loss
            loss = loss_cvl * w_cvl + loss_encoder * w_enc + loss_lpips * w_lpips
            
            # --- Backward Pass ---
            grad, = torch.autograd.grad(loss, [X_adv])
            
            # [關鍵] 梯度 Masking: 只攻擊臉部，背景梯度歸零
            grad = grad * mask 

            # --- Update (Momentum + Gradient Descent) ---
            grad_norm = torch.norm(grad, p=1)
            grad = grad / (grad_norm + 1e-10)
            momentum = momentum + grad
            
            # 使用減法 (Gradient Descent) 來最小化 Loss
            X_adv = X_adv - momentum.sign() * actual_step_size

            # --- Projection ---
            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
            X_adv = torch.clamp(X + delta, min=clamp_min, max=clamp_max)
            
            # [關鍵] 強制還原背景像素 (雙重保險)
            X_adv.data = X_adv.data * mask + X.data * (1 - mask)
            
            X_adv = X_adv.detach()

            # 記錄
            pbar.set_postfix(cvl=loss_cvl.item(), total=loss.item())
            history['total_loss'].append(loss.item())
            history['loss_cvl'].append(loss_cvl.item())
            history['loss_encoder'].append(loss_encoder.item())
            history['loss_lpips'].append(loss_lpips.item())

        if plot_history:
            self._plot_history(history)

        return X_adv, history

    def _plot_history(self, history):
        """
        繪製 Loss 曲線圖並存檔
        """
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

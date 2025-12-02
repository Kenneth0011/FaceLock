import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import dlib
import os
from tqdm import tqdm

# [重要] 強制設定 Matplotlib 後端
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
import matplotlib.pyplot as plt

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
            loss = latent.norm()
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = nn.MSELoss()(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size 

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
# 第二部分：FaceLock 攻擊器 (FaceLockAttacker)
# ==============================================================================

class FaceLockAttacker:
    # ... (init 方法保持不變) ...
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        # ... 省略 ...
        pass

    # ==============================================================================
    # [NEW] 修改了這個函式，增加了 save_debug 參數和繪圖邏輯
    # ==============================================================================
    def get_face_mask(self, image_tensor, pad=50, blur_sigma=20, save_debug=False):
        """
        生成擴大的臉部遮罩 (含額頭延伸)，並可選擇儲存標註特徵點的除錯圖。
        """
        # 將 Tensor 轉為 Numpy (H, W, C), 此時是 RGB 格式
        img_np = image_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()

        # 數值正規化
        if img_np.min() < 0:
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
        elif img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)

        # [NEW] 準備一張用於繪製特徵點的畫布 (複製原圖)
        if save_debug:
            debug_img = img_np.copy()

        dets = self.detector(img_np, 1)
        if len(dets) == 0:
            print("[FaceLock] Warning: No face detected! Returning full mask.")
            return torch.ones_like(image_tensor)

        d = dets[0]

        if self.has_landmarks:
            shape = self.predictor(img_np, d)
            points = []
            for i in range(68):
                px, py = shape.part(i).x, shape.part(i).y
                points.append((px, py))
                # [NEW] 繪製原始 68 個特徵點 (紅色實心圓點)
                if save_debug:
                    cv2.circle(debug_img, (px, py), 3, (255, 0, 0), -1) # RGB 中的紅色

            # 額頭延伸計算
            eyebrow_y_min = min(shape.part(19).y, shape.part(24).y)
            nose_y = shape.part(27).y
            forehead_height = int(abs(nose_y - eyebrow_y_min) * 1.5)

            face_left_x = shape.part(0).x
            face_right_x = shape.part(16).x

            # 定義額頭延伸點
            p_forehead_right = (face_right_x, eyebrow_y_min - forehead_height)
            p_forehead_left = (face_left_x, eyebrow_y_min - forehead_height)

            points.append(p_forehead_right)
            points.append(p_forehead_left)

            # [NEW] 繪製額頭延伸點 (藍色實心圓點)
            if save_debug:
                cv2.circle(debug_img, p_forehead_right, 5, (0, 0, 255), -1) # RGB 中的藍色
                cv2.circle(debug_img, p_forehead_left, 5, (0, 0, 255), -1)

            # 計算凸包 (Convex Hull)
            hull_points = cv2.convexHull(np.array(points))

            # [NEW] 繪製最終的遮罩輪廓 (綠色線條)
            if save_debug:
                cv2.polylines(debug_img, [hull_points], True, (0, 255, 0), 2) # RGB 中的綠色

            # 填滿遮罩
            cv2.fillConvexPoly(mask, hull_points, 1.0)
        else:
            # (省略沒有 landmarks 的情況...)
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            y1_extended = max(0, y1 - pad)
            mask[y1_extended:y2, x1:x2] = 1.0

        # [NEW] 儲存除錯圖片
        if save_debug and self.has_landmarks:
            # OpenCV 儲存圖片需要 BGR 格式，所以要從 RGB 轉 BGR
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            debug_filename = "facelock_landmarks_debug.png"
            cv2.imwrite(debug_filename, debug_img_bgr)
            print(f"[FaceLock] Landmark debug image saved to: {debug_filename}")

        # 遮罩後處理 (膨脹與模糊)
        kernel = np.ones((pad, pad), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)

        mask_tensor = torch.from_numpy(mask).view(1, 1, H, W).repeat(1, 3, 1, 1)
        mask_tensor = mask_tensor.to(image_tensor.device)
        return mask_tensor

    # ... (input_diversity 方法保持不變) ...

    # ==============================================================================
    # 修改 attack 函式，傳遞 save_debug 參數
    # ==============================================================================
    def attack(self, X, model, aligner, fr_model, lpips_fn,
               eps=0.03, step_size=0.01, iters=100,
               clamp_min=-1, clamp_max=1, plot_history=False,
               save_mask=True): # 我們複用這個參數來控制是否輸出除錯圖

        # [NEW] 呼叫 get_face_mask 時，傳入 save_debug=save_mask
        mask = self.get_face_mask(X, pad=50, blur_sigma=20, save_debug=save_mask)

        # (原有的儲存 mask 圖片程式碼可以保留，或合併到上面的 get_face_mask 中)
        # 為了清晰起見，這裡保留你上一版要求的儲存 Mask 功能
        if save_mask:
            mask_vis = mask[0, 0].detach().cpu().numpy()
            mask_vis = (mask_vis * 255).astype(np.uint8)
            cv2.imwrite("facelock_mask_debug.png", mask_vis)
            # print(f"[FaceLock] Mask image saved to: facelock_mask_debug.png") # 上面已經有 print 了，這裡可以註解掉

        # ... (後續攻擊程式碼保持不變) ...
        noise = (torch.rand(*X.shape) * 2 * eps - eps).to(X.device)
        # ...
        return X_adv, history
    # ... (其餘方法保持不變) ...
    def _plot_history(self, history):
        """
        繪製 Loss 曲線圖，標題與格式完全依照使用者要求
        """
        print("Plotting losses...")
        plt.figure(figsize=(12, 10))
        plt.suptitle('FaceLock Loss Convergence', fontsize=16)

        # 圖 1: Total Loss
        plt.subplot(2, 2, 1)
        plt.plot(history['total_loss'])
        plt.title('Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

        # 圖 2: Face Recognition Score (loss_cvl)
        plt.subplot(2, 2, 2)
        plt.plot(history['loss_cvl'], color='red')
        plt.title('Face Recognition Score (loss_cvl)')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.grid(True)

        # 圖 3: Encoder MSE Loss (loss_encoder)
        plt.subplot(2, 2, 3)
        plt.plot(history['loss_encoder'], color='green')
        plt.title('Encoder MSE Loss (loss_encoder)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

        # 圖 4: Perceptual Similarity (loss_lpips)
        plt.subplot(2, 2, 4)
        plt.plot(history['loss_lpips'], color='purple')
        plt.title('Perceptual Similarity (loss_lpips)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = "facelock_loss_convergence.png"
        plt.savefig(save_path)
        print(f"Loss plot saved to: {save_path}")
        plt.close()

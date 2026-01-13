import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
from utils import compute_score

# 設定 Matplotlib 後端
os.environ['MPLBACKEND'] = 'Agg'

# ==========================================
# [新增模組] Stable Diffusion Attention 攔截器
# 這是實現 "Structure Disruption" 的核心
# ==========================================
class AttentionStore:
    def __init__(self):
        self.current_attention = [] 
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 我們只攔截 Self-Attention (決定結構的部分)
        # 並且只針對 'up' (解碼階段) 和 'mid' (中間層)，這些對結構影響最大
        if not is_cross and place_in_unet in ["mid", "up"]:
            self.current_attention.append(attn)

    def reset(self):
        self.current_attention = []

class AttackAttentionProcessor:
    def __init__(self, store, place_in_unet):
        self.store = store
        self.place = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        # 判斷是否為 Cross Attention
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # [關鍵] 將 Attention Map 存起來
        self.store(attention_probs, is_cross, self.place)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def register_attention_control(unet, store):
    """將 Hook 註冊到 U-Net 中"""
    def register_rec(net, place_in_unet):
        if net.__class__.__name__ == 'Attention':
            net.set_processor(AttackAttentionProcessor(store, place_in_unet))
        elif hasattr(net, 'children'):
            for name, child in net.named_children():
                new_place = place_in_unet
                if "mid" in name: new_place = "mid"
                elif "up" in name: new_place = "up"
                elif "down" in name: new_place = "down"
                register_rec(child, new_place)
    register_rec(unet, "base")

# ==========================================
# [保留] DIM 輔助函式 (維持不變)
# ==========================================
def input_diversity(x, resize_rate=0.9, diversity_prob=0.7):
    if torch.rand(1) > diversity_prob:
        return x
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_resize, high=img_size, size=(1,)).item()
    rescaled = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    h_rem = img_size - rnd
    w_rem = img_size - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,)).item()
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    return padded

# ==========================================
# [修改] 繪圖函數：新增 Structure Loss 曲線
# ==========================================
def plot_facelock_history(history):
    print("Plotting losses...")
    
    # 固定使用 2x2 版面，顯示四種資訊
    plt.figure(figsize=(14, 10))
    
    # 1. Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # 2. FR Score (Identity)
    plt.subplot(2, 2, 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (loss_cvl)')
    plt.xlabel('Iteration')
    plt.ylabel('Score (Lower is Better)')
    plt.grid(True)

    # 3. Encoder MSE (Original Feature Disruption)
    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('VAE Latent MSE (Basic Disruption)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Higher is Better)')
    plt.grid(True)

    # 4. [新增] Structure Loss (Visual Disruption)
    # 如果 history 中有這個鍵值才畫
    if 'loss_structure' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['loss_structure'], color='purple')
        plt.title('Structure Attention Loss (Visual Privacy)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Higher is Better)') # 我們希望差異越大越好
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('FaceLock + Structure Disruption Convergence', fontsize=16)
    
    save_path = "facelock_structure_convergence.png"
    plt.savefig(save_path) 
    print(f"Loss plot saved to: {save_path}")
    plt.close()

# ==========================================
# [核心修改] Facelock 主函式
# 加入 sd_pipe 參數與 Structure Loss 計算邏輯
# ==========================================
def facelock(X, model, aligner, fr_model, lpips_fn=None, 
             sd_pipe=None,              # [新增] 傳入 Stable Diffusion Pipeline
             eps=0.07, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot_history=True, tv_weight=0): 

    # 初始化對抗樣本
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device)
    X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max).half() # 假設用 float16 節省記憶體
    X_adv.requires_grad = True

    # 準備 VAE 與 Reference
    vae = model.vae
    clean_latent = vae.encode(X).latent_dist.mean.detach()

    # [新增] 準備 Structure Disruption (如果 sd_pipe 有傳入)
    use_structure_attack = sd_pipe is not None
    clean_attentions = []
    attention_store = None
    target_timestep = None
    
    if use_structure_attack:
        print(">>> 啟動 Structure Disruption (視覺隱私保護模式)")
        attention_store = AttentionStore()
        register_attention_control(sd_pipe.unet, attention_store)
        
        # 設定攻擊的時間步 t (論文建議 t=500 左右最有效)
        target_timestep = torch.tensor([500]).to(X.device)
        
        # 預先計算原圖的 Attention Map (Clean Structure)
        with torch.no_grad():
            # 需確保輸入範圍適配 SD [-1, 1]
            # 假設 X 已經是正規化過的，這裡視情況調整
            latents_clean = sd_pipe.vae.encode(X).latent_dist.sample() * 0.18215
            
            # 清空 store 並執行一次 U-Net forward
            attention_store.reset()
            # 傳入空 prompt embedding
            empty_embeds = sd_pipe._encode_prompt("", X.device, 1, False, None)[0]
            _ = sd_pipe.unet(latents_clean, target_timestep, encoder_hidden_states=empty_embeds)
            
            # 深拷貝存起來
            clean_attentions = copy.deepcopy(attention_store.current_attention)
            print(f"    已捕捉 {len(clean_attentions)} 層 Attention Map 作為結構基準")

    # 記錄 Loss
    history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': [], 'loss_structure': []}
    pbar = tqdm(range(iters))

    for i in pbar:
        # 1. VAE 重建路徑
        latent = vae.encode(X_adv).latent_dist.mean
        image_recon = vae.decode(latent).sample.clip(-1, 1)

        # 2. DIM (多樣化輸入)
        image_dim = input_diversity(image_recon, resize_rate=0.9, diversity_prob=0.7)

        # ----------------------------------------
        # 計算各項 Loss
        # ----------------------------------------
        
        # (A) FR Score (身份識別) -> 越低越好
        loss_cvl = compute_score(image_dim.float(), X.float(), aligner=aligner, fr_model=fr_model)
        
        # (B) VAE MSE (基礎特徵) -> 越大越好
        loss_encoder = F.mse_loss(latent, clean_latent)
        
        # (C) Structure Loss (結構破壞) -> 越大越好 (與原圖結構差異越大)
        loss_structure = torch.tensor(0.0).to(X.device)
        
        if use_structure_attack:
            # 計算現在的 Attention Map
            # 注意：這裡需要 gradient，所以不能用 no_grad
            latents_adv = sd_pipe.vae.encode(X_adv).latent_dist.sample() * 0.18215
            
            attention_store.reset()
            empty_embeds = sd_pipe._encode_prompt("", X.device, 1, False, None)[0]
            _ = sd_pipe.unet(latents_adv, target_timestep, encoder_hidden_states=empty_embeds)
            
            adv_attentions = attention_store.current_attention
            
            # 計算與 Clean Attention 的差異 (MSE)
            # 我們只取部分層計算以節省記憶體，或者計算平均
            layer_losses = []
            for clean_map, adv_map in zip(clean_attentions, adv_attentions):
                # Maximize distance => Minimize negative MSE
                layer_losses.append(F.mse_loss(adv_map, clean_map))
            
            if layer_losses:
                loss_structure = torch.stack(layer_losses).mean()

        # ----------------------------------------
        # 總 Loss 加權
        # ----------------------------------------
        # 權重建議：
        # FR: 5.0 (保證機器防禦)
        # Encoder: 1.0 (基礎干擾)
        # Structure: 20.0 (因為 Attention數值很小，需要較大權重來推動結構改變)
        
        total_loss = -loss_cvl * 5.0 + loss_encoder * 1.0 
        
        if use_structure_attack:
            # 我們希望結構差異(loss_structure)越大越好 -> 所以加號放入 total loss (因為是梯度上升)
            # 或者是： total_loss = ... - loss_structure * weight (如果是梯度下降法)
            # 根據您原本的代碼邏輯：
            # grad = autograd.grad(loss, [X_adv])
            # X_adv = X_adv + step_size * grad.sign() (這是梯度上升 Gradient Ascent)
            # 所以我們希望 Total Loss 越大越好
            total_loss += loss_structure * 20.0

        # 計算梯度
        grad, = torch.autograd.grad(total_loss, [X_adv])
        
        # 更新圖片 (PGD Attack)
        X_adv.data = X_adv.data + step_size * grad.sign()
        
        # 限制範圍
        X_adv.data = torch.max(torch.min(X_adv.data, X + eps), X - eps)
        X_adv.data = torch.clamp(X_adv.data, min=clamp_min, max=clamp_max)

        # 記錄與顯示
        history['total_loss'].append(total_loss.item())
        history['loss_cvl'].append(loss_cvl.item())
        history['loss_encoder'].append(loss_encoder.item())
        if use_structure_attack:
            history['loss_structure'].append(loss_structure.item())
            
        pbar.set_postfix(
            cvl=f"{loss_cvl.item():.2f}", 
            struc=f"{loss_structure.item():.4f}" if use_structure_attack else "N/A"
        )

    if plot_history:
        try:
            plot_facelock_history(history)
        except Exception as e:
            print(f"Plotting failed: {e}")

    return X_adv, history

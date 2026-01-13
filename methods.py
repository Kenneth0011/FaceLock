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
# 1. 基礎攻擊函式 (CW, Encoder, VAE)
# ==========================================

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

    history = {'total_loss': [], 'loss1': [], 'loss2': []} 

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        history['total_loss'].append(cost.item())
        history['loss1'].append(loss1.item())
        history['loss2'].append(loss2.item())
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv, history 

# Encoder attack
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean
    pbar = tqdm(range(iters))
    
    history = {'loss': []} 
    
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

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        history['loss'].append(loss.item()) 

    return X_adv, history 

# VAE attack
def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    
    history = {'loss': []} 
    
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None
        
        history['loss'].append(loss.item()) 

    return X_adv, history 


# ==========================================
# 2. Structure Disruption 相關模組 (增強穩健版)
# ==========================================
class AttentionStore:
    def __init__(self):
        self.current_attention = [] 
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 攔截 'mid' 和 'up' 層的 Self-Attention
        if not is_cross and place_in_unet in ["mid", "up"]:
            self.current_attention.append(attn)

    def reset(self):
        self.current_attention = []

class AttackAttentionProcessor:
    def __init__(self, store, place_in_unet):
        self.store = store
        self.place = place_in_unet

    def _batch_to_head_dim(self, tensor, heads):
        # [關鍵修正]：如果 tensor 是 2D [Batch, Dim]，我們將其視為 [Batch, 1, Dim]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)

        batch_size, seq_len, dim = tensor.shape
        head_dim = dim // heads
        tensor = tensor.reshape(batch_size, seq_len, heads, head_dim)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(batch_size * heads, seq_len, head_dim)

    def _head_to_batch_dim(self, tensor, heads):
        batch_value, seq_len, head_dim = tensor.shape
        batch_size = batch_value // heads
        tensor = tensor.reshape(batch_size, heads, seq_len, head_dim)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(batch_size, seq_len, heads * head_dim)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # 使用手動 Reshape 替代 attn.head_to_batch_dim
        query = self._batch_to_head_dim(query, attn.heads)
        key = self._batch_to_head_dim(key, attn.heads)
        value = self._batch_to_head_dim(value, attn.heads)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 儲存 Attention Map
        self.store(attention_probs, is_cross, self.place)
        
        hidden_states = torch.bmm(attention_probs, value)
        
        # 手動轉回 batch dim
        hidden_states = self._head_to_batch_dim(hidden_states, attn.heads)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def register_attention_control(unet, store):
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

# DIM (Diverse Inputs Method)
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

# 繪圖函式
def plot_facelock_history(history):
    print("Plotting losses...")
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['loss_cvl'], color='red')
    plt.title('Face Recognition Score (loss_cvl)')
    plt.xlabel('Iteration')
    plt.ylabel('Score (Lower is Better)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['loss_encoder'], color='green')
    plt.title('VAE Latent MSE (Basic Disruption)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Higher is Better)')
    plt.grid(True)

    if 'loss_structure' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['loss_structure'], color='purple')
        plt.title('Structure Attention Loss (Visual Privacy)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Higher is Better)')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('FaceLock + Structure Disruption Convergence', fontsize=16)
    
    save_path = "facelock_structure_convergence.png"
    plt.savefig(save_path) 
    print(f"Loss plot saved to: {save_path}")
    plt.close()

# 輔助函式：處理 U-Net 輸入通道不匹配問題
def prepare_unet_input(unet, latents):
    in_channels = unet.config.in_channels
    if in_channels == 8 and latents.shape[1] == 4:
        return torch.cat([latents, latents], dim=1)
    return latents

# ==========================================
# 3. Facelock 主函式 (Structure Disruption 版)
# ==========================================
def facelock(X, model, aligner, fr_model, lpips_fn=None, 
             sd_pipe=None,
             eps=0.07, step_size=0.01, iters=100, 
             clamp_min=-1, clamp_max=1, 
             plot_history=True, tv_weight=0): 

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device)
    X_adv = torch.clamp(X_adv, min=clamp_min, max=clamp_max).half()
    X_adv.requires_grad = True

    vae = model.vae
    clean_latent = vae.encode(X).latent_dist.mean.detach()

    use_structure_attack = sd_pipe is not None
    clean_attentions = []
    attention_store = None
    target_timestep = None
    
    if use_structure_attack:
        print(">>> 啟動 Structure Disruption (視覺隱私保護模式)")
        attention_store = AttentionStore()
        register_attention_control(sd_pipe.unet, attention_store)
        
        target_timestep = torch.tensor([500]).to(X.device)
        
        with torch.no_grad():
            latents_clean = sd_pipe.vae.encode(X).latent_dist.sample() * 0.18215
            attention_store.reset()
            empty_embeds = sd_pipe._encode_prompt("", X.device, 1, False, None)[0]
            
            unet_input = prepare_unet_input(sd_pipe.unet, latents_clean)
            
            _ = sd_pipe.unet(unet_input, target_timestep, encoder_hidden_states=empty_embeds)
            clean_attentions = copy.deepcopy(attention_store.current_attention)
            print(f"    已捕捉 {len(clean_attentions)} 層 Attention Map 作為結構基準")

    history = {'total_loss': [], 'loss_cvl': [], 'loss_encoder': [], 'loss_structure': []}
    pbar = tqdm(range(iters))

    for i in pbar:
        # VAE Path
        latent = vae.encode(X_adv).latent_dist.mean
        image_recon = vae.decode(latent).sample.clip(-1, 1)

        # DIM
        image_dim = input_diversity(image_recon, resize_rate=0.9, diversity_prob=0.7)
        
        # Loss Calc
        loss_cvl = compute_score(image_dim.float(), X.float(), aligner=aligner, fr_model=fr_model)
        loss_encoder = F.mse_loss(latent, clean_latent)
        
        loss_structure = torch.tensor(0.0).to(X.device)
        if use_structure_attack:
            latents_adv = sd_pipe.vae.encode(X_adv).latent_dist.sample() * 0.18215
            attention_store.reset()
            empty_embeds = sd_pipe._encode_prompt("", X.device, 1, False, None)[0]
            
            unet_input_adv = prepare_unet_input(sd_pipe.unet, latents_adv)
            
            _ = sd_pipe.unet(unet_input_adv, target_timestep, encoder_hidden_states=empty_embeds)
            
            adv_attentions = attention_store.current_attention
            
            layer_losses = []
            for clean_map, adv_map in zip(clean_attentions, adv_attentions):
                layer_losses.append(F.mse_loss(adv_map, clean_map))
            
            if layer_losses:
                loss_structure = torch.stack(layer_losses).mean()

        # 總 Loss 計算
        total_loss = -loss_cvl * 5.0 + loss_encoder * 1.0 
        
        if use_structure_attack:
            total_loss += loss_structure * 20.0

        grad, = torch.autograd.grad(total_loss, [X_adv])
        X_adv.data = X_adv.data + step_size * grad.sign()
        
        X_adv.data = torch.max(torch.min(X_adv.data, X + eps), X - eps)
        X_adv.data = torch.clamp(X_adv.data, min=clamp_min, max=clamp_max)

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

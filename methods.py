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

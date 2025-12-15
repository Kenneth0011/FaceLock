import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision
import lpips
import gc # [新增] 垃圾回收
import pdb

# --- Import 自定義模組 ---
from utils import load_model_by_repo_id, pil_to_input
from methods import cw_l2_attack, vae_attack, encoder_attack, facelock

# --- Import Diffusers ---
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

def get_args_parser():
    parser = argparse.ArgumentParser()

    # 1. Image arguments
    parser.add_argument("--input_path", required=True, type=str, help="path to the input image")

    # 2. Target model arguments
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="[instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="huggingface model id")
    
    # 3. Attack arguments
    parser.add_argument("--defend_method", required=True, type=str, help="[encoder/vae/cw/facelock]")
    parser.add_argument("--attack_budget", default=0.03, type=float, help="attack budget (eps)")
    parser.add_argument("--step_size", default=0.01, type=float, help="attack step size")
    parser.add_argument("--num_iters", default=100, type=int, help="number of iterations")
    parser.add_argument("--targeted", default=True, action='store_true', help="targeted attack")
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="untargeted attack")

    # 3.1 CW attack specific
    parser.add_argument("--c", default=0.03, type=float, help="constant ratio for cw attack")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate for cw attack")

    # 4. Output & Logging arguments
    parser.add_argument("--output_path", default=None, type=str, help="path to save the protected image")
    parser.add_argument("--plot_history", action='store_true', help="save the loss convergence plot")

    return parser

def process_encoder_attack(X, model, args):
    with torch.autocast("cuda"):
        result = encoder_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
            targeted=args.targeted,
        )
        X_adv = result[0] if isinstance(result, tuple) else result
    return X_adv

def process_vae_attack(X, model, args):
    with torch.autocast("cuda"):
        result = vae_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
        )
        X_adv = result[0] if isinstance(result, tuple) else result
    return X_adv

def process_cw_attack(X, model, args):
    result = cw_l2_attack(
        X=X,
        model=model,
        c=args.c,
        lr=args.lr,
        iters=args.num_iters,
    )
    X_adv = result[0] if isinstance(result, tuple) else result
    delta = X_adv - X
    delta_clip = delta.clip(-args.attack_budget, args.attack_budget)
    X_adv = (X + delta_clip).clip(0, 1)
    return X_adv

def process_facelock(X, model, args):
    fr_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
    aligner_id = 'minchul/cvlface_DFA_mobilenet'
    device = 'cuda'
    
    print(f"Loading FR model: {fr_id}...")
    fr_model = load_model_by_repo_id(repo_id=fr_id,
                                     save_path=f'{os.environ["HF_HOME"]}/{fr_id}',
                                     HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN', None)).to(device)
    aligner = load_model_by_repo_id(repo_id=aligner_id,
                                    save_path=f'{os.environ["HF_HOME"]}/{aligner_id}',
                                    HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN', None)).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    with torch.autocast("cuda"):
        result = facelock_robust(
            X=X,
            model=model,
            aligner=aligner,
            fr_model=fr_model,
            lpips_fn=lpips_fn,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
            decay=1.0,         
            noise_std=0.01,    
            plot_history=args.plot_history
        )
        X_adv = result[0] 

    # [Memory Fix] 清理這些輔助模型以釋放記憶體
    del fr_model, aligner, lpips_fn
    torch.cuda.empty_cache()
    
    return X_adv

def main(args):
    # 0. 檢查輸出路徑
    if args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # 1. Prepare image
    print(f"Loading image from {args.input_path}...")
    init_image = Image.open(args.input_path).convert("RGB")
    
    if args.defend_method != "cw":
        X = pil_to_input(init_image).cuda().half()
    else:
        to_tensor = torchvision.transforms.ToTensor()
        X = to_tensor(init_image).cuda().unsqueeze(0)

    # 2. Prepare Target Model
    print(f"Loading target model: {args.target_model} ({args.model_id})...")
    model = None
    dtype = torch.float16 if args.defend_method != "cw" else torch.float32

    if args.target_model == "stable-diffusion":
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
    elif args.target_model == "instruct-pix2pix":
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    else:
        raise ValueError(f"Invalid target_model '{args.target_model}'.")
    
    model.to("cuda")

    # ---------------------------------------------------------
    # [Memory Fix] 記憶體優化關鍵步驟
    # ---------------------------------------------------------
    if args.defend_method in ["facelock", "vae", "encoder"]:
        print("⚡ Optimizing memory: Offloading UNet/TextEncoder to CPU...")
        
        # 1. 將不需參與攻擊計算的模組移至 CPU
        if hasattr(model, "unet"):
            model.unet = model.unet.to("cpu")
        if hasattr(model, "text_encoder"):
            model.text_encoder = model.text_encoder.to("cpu")
        if hasattr(model, "safety_checker") and model.safety_checker is not None:
            model.safety_checker = model.safety_checker.to("cpu")
            
        # 2. 強制進行垃圾回收與顯存清理
        gc.collect()
        torch.cuda.empty_cache()

        # 3. 啟用 VAE Slicing/Tiling (這能大幅降低 decode 時的顯存峰值)
        if hasattr(model, "vae"):
            print("⚡ Enabling VAE Slicing & Tiling for memory efficiency...")
            model.vae.enable_slicing()
            try:
                model.vae.enable_tiling()
            except AttributeError:
                pass # 舊版 diffusers 可能沒有 enable_tiling
    # ---------------------------------------------------------

    # 3. Select Defense Method
    defend_fn = None
    if args.defend_method == "encoder":
        defend_fn = process_encoder_attack
    elif args.defend_method == "vae":
        defend_fn = process_vae_attack
    elif args.defend_method == "cw":
        defend_fn = process_cw_attack
    elif args.defend_method == "facelock":
        defend_fn = process_facelock
    else:
        raise ValueError(f"Invalid defend_method '{args.defend_method}'.")
    
    # 4. Run Defense
    print(f"Running defense: {args.defend_method} (iters={args.num_iters}, budget={args.attack_budget})...")
    X_adv = defend_fn(X, model, args)

    # 5. Save Output
    to_pil = torchvision.transforms.ToPILImage()
    
    if args.defend_method != "cw":
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    
    if len(X_adv.shape) == 4:
        X_adv = X_adv[0]

    protected_image = to_pil(X_adv).convert("RGB")
    
    if args.output_path:
        protected_image.save(args.output_path)
        print(f"✅ Protected image saved to: {args.output_path}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

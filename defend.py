import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision
import lpips
import pdb

# 引入 Hugging Face Diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

# 引入本地模組
from utils import load_model_by_repo_id, pil_to_input
# 確保 methods.py 是最新版 (Anti-Inpainting 版本)
from methods import cw_l2_attack, vae_attack, encoder_attack, facelock

def get_args_parser():
    parser = argparse.ArgumentParser(description="Image Defense Script")

    # 1. Image arguments
    parser.add_argument("--input_path", required=True, type=str, help="Path to the image you hope to protect")

    # 2. Target model arguments
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="Target model: [instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="Hugging Face model ID")
    
    # 3. Attack (Defense) arguments
    parser.add_argument("--defend_method", required=True, type=str, help="Method: [encoder/vae/cw/facelock]")
    parser.add_argument("--attack_budget", default=0.03, type=float, help="Attack budget (epsilon)")
    parser.add_argument("--step_size", default=0.01, type=float, help="Attack step size")
    parser.add_argument("--num_iters", default=100, type=int, help="Number of iterations")
    parser.add_argument("--targeted", default=True, action='store_true', help="Use targeted attack")
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="Use untargeted attack")
    
    # 繪圖開關
    parser.add_argument("--plot", action='store_true', help="Enable plotting of loss convergence and mask visualization")

    # 3.1 CW attack specific arguments
    parser.add_argument("--c", default=0.03, type=float, help="Constant ratio for CW attack")
    parser.add_argument("--lr", default=0.03, type=float, help="Learning rate for CW attack")

    # 4. Output arguments
    parser.add_argument("--output_path", default="output.png", type=str, help="Path to save the protected image")
    
    return parser

# ... (Encoder/VAE/CW attack 函數保持不變) ...
def process_encoder_attack(X, model, args):
    print(f"Running Encoder Attack (eps={args.attack_budget}, iters={args.num_iters})...")
    with torch.autocast("cuda"):
        X_adv = encoder_attack(
            X=X, model=model, eps=args.attack_budget, step_size=args.step_size,
            iters=args.num_iters, clamp_min=-1, clamp_max=1, targeted=args.targeted,
        )
    return X_adv

def process_vae_attack(X, model, args):
    print(f"Running VAE Attack (eps={args.attack_budget}, iters={args.num_iters})...")
    with torch.autocast("cuda"):
        X_adv = vae_attack(
            X=X, model=model, eps=args.attack_budget, step_size=args.step_size,
            iters=args.num_iters, clamp_min=-1, clamp_max=1,
        )
    return X_adv

def process_cw_attack(X, model, args):
    print(f"Running CW L2 Attack (c={args.c}, lr={args.lr}, iters={args.num_iters})...")
    X_adv = cw_l2_attack(X=X, model=model, c=args.c, lr=args.lr, iters=args.num_iters)
    delta = X_adv - X
    delta_clip = delta.clip(-args.attack_budget, args.attack_budget)
    X_adv = (X + delta_clip).clip(0, 1)
    return X_adv

# =========================================================
# [修改重點] 輕量化的 Facelock 處理流程
# =========================================================
def process_facelock(X, model, args):
    print(f"Running Anti-Inpainting Facelock (eps={args.attack_budget}, iters={args.num_iters})...")
    
    # [移除] 不再載入 ArcFace 和 Aligner 模型，節省顯存
    device = 'cuda'
    
    # 只需要 LPIPS 來維持畫質
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    with torch.autocast("cuda"):
        X_adv = facelock(
            X=X,
            model=model,
            aligner=None,   # [修改] 傳入 None 即可
            fr_model=None,  # [修改] 傳入 None 即可
            lpips_fn=lpips_fn,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
            plot=args.plot
        )
    return X_adv

def main(args):
    # 0. Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    # 1. Prepare the image
    print(f"Loading image from {args.input_path}...")
    init_image = Image.open(args.input_path).convert("RGB")
    
    to_tensor = torchvision.transforms.ToTensor()
    if args.defend_method != "cw":
        X = pil_to_input(init_image).cuda().half()
    else:
        X = to_tensor(init_image).cuda().unsqueeze(0)

    # 2. Prepare the targeted model
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

    # 3. Set up defense method
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
    
    # 4. Process defense
    X_adv = defend_fn(X, model, args)

    # 5. Convert back to image and save
    print(f"Saving protected image to {args.output_path}...")
    to_pil = torchvision.transforms.ToPILImage()
    
    if args.defend_method != "cw":
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    
    if X_adv.dim() == 4:
        X_adv = X_adv[0]
        
    protected_image = to_pil(X_adv.float()).convert("RGB")
    protected_image.save(args.output_path)
    print("Done!")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision
import lpips
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

# 假設 utils.py 已經存在
from utils import load_model_by_repo_id, pil_to_input

# 引入 methods.py 裡的攻擊方法與 FaceLockAttacker 類別
from methods import cw_l2_attack, vae_attack, encoder_attack, FaceLockAttacker 

def get_args_parser():
    parser = argparse.ArgumentParser()

    # 1. Image arguments
    parser.add_argument("--input_path", required=True, type=str, help="path to the input image")

    # 2. Target model arguments
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="target model [instruct-pix2pix/stable-diffusion]")
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="huggingface model id")
    
    # 3. Attack arguments
    parser.add_argument("--defend_method", required=True, type=str, help="chosen method [encoder/vae/cw/facelock]")
    parser.add_argument("--attack_budget", default=0.03, type=float, help="perturbation budget (eps)")
    parser.add_argument("--step_size", default=0.01, type=float, help="attack step size")
    parser.add_argument("--num_iters", default=100, type=int, help="number of iterations")
    parser.add_argument("--targeted", default=True, action='store_true', help="use targeted attack")
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="use untargeted attack")

    # 3.1 CW attack arguments
    parser.add_argument("--c", default=0.03, type=float, help="constant for cw attack")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate for cw attack")

    # 3.2 FaceLock arguments (Mask & Plot)
    parser.add_argument("--plot_history", action='store_true', help="plot loss history (facelock only)")
    parser.add_argument("--dlib_path", default="shape_predictor_68_face_landmarks.dat", type=str, help="path to dlib predictor")

    # 4. Output arguments
    parser.add_argument("--output_path", default=None, type=str, help="path to save protected image")
    
    return parser

# ==============================================================================
# Wrappers
# ==============================================================================

def process_encoder_attack(X, model, args):
    with torch.autocast("cuda"):
        X_adv, history = encoder_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
            targeted=args.targeted,
        )
    return X_adv, history

def process_vae_attack(X, model, args):
    with torch.autocast("cuda"):
        X_adv, history = vae_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
        )
    return X_adv, history

def process_cw_attack(X, model, args):
    X_adv, history = cw_l2_attack(
        X=X,
        model=model,
        c=args.c,
        lr=args.lr,
        iters=args.num_iters,
    )
    delta = X_adv - X
    delta_clip = delta.clip(-args.attack_budget, args.attack_budget)
    X_adv = (X + delta_clip).clip(0, 1)
    return X_adv, history

def process_facelock(X, model, args):
    # Load auxiliary models
    fr_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
    aligner_id = 'minchul/cvlface_DFA_mobilenet'
    device = 'cuda'

    print("[Defend] Loading FaceLock auxiliary models...")
    fr_model = load_model_by_repo_id(repo_id=fr_id,
                                     save_path=f'{os.environ["HF_HOME"]}/{fr_id}',
                                     HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN')).to(device)
    aligner = load_model_by_repo_id(repo_id=aligner_id,
                                    save_path=f'{os.environ["HF_HOME"]}/{aligner_id}',
                                    HF_TOKEN=os.environ.get('HUGGINGFACE_HUB_TOKEN')).to(device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    with torch.autocast("cuda"):
        # Instantiate the Attacker Class
        attacker = FaceLockAttacker(predictor_path=args.dlib_path)
        
        # Run Attack
        X_adv, history = attacker.attack(
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
            plot_history=args.plot_history
        )
        
    return X_adv, history

# ==============================================================================
# Main
# ==============================================================================

def main(args):
    print(f"Loading image: {args.input_path}")
    init_image = Image.open(args.input_path).convert("RGB")
    to_tensor = torchvision.transforms.ToTensor()
    
    if args.defend_method != "cw":
        X = pil_to_input(init_image).cuda().half()
    else:
        X = to_tensor(init_image).cuda().unsqueeze(0)

    print(f"Loading Target Model: {args.target_model}...")
    dtype = torch.float16 if args.defend_method != "cw" else torch.float32
    
    if args.target_model == "stable-diffusion":
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_id, torch_dtype=dtype, safety_checker=None
        )
    elif args.target_model == "instruct-pix2pix":
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.model_id, torch_dtype=dtype, safety_checker=None
        )
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    else:
        raise ValueError(f"Unknown target_model: {args.target_model}")
    
    model.to("cuda")

    # Select Method
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
        raise ValueError(f"Unknown defend_method: {args.defend_method}")
    
    print(f"Running Defend: {args.defend_method}...")
    X_adv, history = defend_fn(X, model, args)
    
    if history is not None:
        print("Defend complete.")

    # Save
    to_pil = torchvision.transforms.ToPILImage()
    if args.defend_method != "cw":
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1)
    else:
        X_adv = X_adv.clamp(0, 1)

    protected_image = to_pil(X_adv[0]).convert("RGB")
    
    if args.output_path:
        protected_image.save(args.output_path)
        print(f"Saved to: {args.output_path}")
    else:
        print("Warning: No output path provided.")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_{args.defend_method}{ext}"
        print(f"Output defaults to: {args.output_path}")

    main(args)

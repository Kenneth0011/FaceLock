import os
from PIL import Image
import numpy as np
import torch
import torchvision
import lpips
from utils import load_model_by_repo_id, pil_to_input #
from methods import cw_l2_attack, vae_attack, encoder_attack, facelock #
import argparse
from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import pdb
import contextlib # 導入 contextlib 來處理 autocast

def get_args_parser():
    parser = argparse.ArgumentParser()

    # 1. image arguments
    # 修正: 'parser.add.argument' -> 'parser.add_argument'
    parser.add_argument("--input_path", required=True, type=str, help="the path to the image you hope to protect") #

    # 2. target model arguments
    # 修正: 'parser.add.argument' -> 'parser.add_argument'
    parser.add_argument("--target_model", default="instruct-pix2pix", type=str, help="the target image editing model [instruct-pix2pix/stable-diffusion]") #
    parser.add_argument("--model_id", default="timbrooks/instruct-pix2pix", type=str, help="model id from hugging face for the model") #
    
    # 3. attack arguments
    # 修正: 'parser.add.argument' -> 'parser.add_argument'
    parser.add_argument("--defend_method", required=True, type=str, help="the chosen attack method between [encoder/vae/cw/facelock]") #
    parser.add_argument("--attack_budget", default=0.03, type=float, help="the attack budget") #
    parser.add_argument("--step_size", default=0.01, type=float, help="the attack step size") #
    parser.add_argument("--num_iters", default=100, type=int, help="the number of attack iterations") #
    parser.add_argument("--targeted", default=True, action='store_true', help="targeted (towards 0 tensor) attack") #
    parser.add_argument("--untargeted", action='store_false', dest='targeted', help="untargeted attack") #

    # 3.1 cw attack other arguments
    # 修正: 'parser.add.argument' -> 'parser.add_argument'
    parser.add_argument("--c", default=0.03, type=float, help="the constant ratio used in cw attack") #
    parser.add_argument("--lr", default=0.03, type=float, help="the learning rate for the optimizer used in cw attack") #

    # 4. output arguments
    # 修正: 'parser.add.argument' -> 'parser.add_argument'
    parser.add_argument("--output_path", default=None, type=str, help="the output path the protected images") #
    return parser

def process_encoder_attack(X, model, args, device): # 傳入 device
    # [CPU 支援] 僅在 CUDA 上使用 autocast
    context_manager = torch.autocast(device_type=device) if device == 'cuda' else contextlib.nullcontext()
    with context_manager:
        X_adv = encoder_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
            targeted=args.targeted,
        ) #
    return X_adv

def process_vae_attack(X, model, args, device): # 傳入 device
    # [CPU 支援] 僅在 CUDA 上使用 autocast
    context_manager = torch.autocast(device_type=device) if device == 'cuda' else contextlib.nullcontext()
    with context_manager:
        X_adv = vae_attack(
            X=X,
            model=model,
            eps=args.attack_budget,
            step_size=args.step_size,
            iters=args.num_iters,
            clamp_min=-1,
            clamp_max=1,
        ) #
    return X_adv

def process_cw_attack(X, model, args, device): # 傳入 device
    X_adv = cw_l2_attack(
        X=X,
        model=model,
        c=args.c,
        lr=args.lr,
        iters=args.num_iters,
    ) #
    delta = X_adv - X
    delta_clip = delta.clip(-args.attack_budget, args.attack_budget)
    X_adv = (X + delta_clip).clip(0, 1)
    return X_adv

def process_facelock(X, model, args, device): # 傳入 device
    fr_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m' #
    aligner_id = 'minchul/cvlface_DFA_mobilenet' #

    # --- [KeyError 修正] 開始 ---

    # 1. 安全地獲取 HF_HOME，如果未設定，則使用預設的快取路徑
    #    os.environ.get() 在找不到時會返回 None，而不是報錯
    hf_home_path = os.environ.get("HF_HOME")
    if hf_home_path is None:
        # 使用 Hugging Face 的標準預設路徑 (e.g., /home/chlin/.cache/huggingface)
        hf_home_path = os.path.expanduser("~/.cache/huggingface") 
        print(f"[process_facelock] 警告: 環境變數 'HF_HOME' 未設定。")
        print(f"[process_facelock] 將使用預設路徑: {hf_home_path}")

    # 2. 安全地獲取 HF_TOKEN，如果未設定，則為 None
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
         print(f"[process_facelock] 警告: 'HUGGINGFACE_HUB_TOKEN' 未設定。僅能下載公開模型。")

    # 3. 使用修正後的變數 (hf_home_path 和 hf_token) 來加載模型
    fr_model = load_model_by_repo_id(repo_id=fr_id,
                                    save_path=f'{hf_home_path}/{fr_id}',
                                    HF_TOKEN=hf_token).to(device) #
    aligner = load_model_by_repo_id(repo_id=aligner_id,
                                    save_path=f'{hf_home_path}/{aligner_id}',
                                    HF_TOKEN=hf_token).to(device) #
    
    # --- [KeyError 修正] 結束 ---
    
    lpips_fn = lpips.LPIPS(net="vgg").to(device) #

    # [CPU 支援] 僅在 CUDA 上使用 autocast
    context_manager = torch.autocast(device_type=device) if device == 'cuda' else contextlib.nullcontext()
    with context_manager:
        X_adv = facelock(
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
        ) #
    return X_adv

def main(args):
    # --- 1. 準備設備和數據類型 [CPU 支援] ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{__name__}] Using device: {device}")
    
    # CPU 不支援 float16，CW 攻擊需要 float32
    if device == 'cpu' or args.defend_method == "cw":
        dtype = torch.float32
    else:
        dtype = torch.float16
    print(f"[{__name__}] Using dtype: {dtype}")

    # --- 2. 準備圖片 ---
    init_image = Image.open(args.input_path).convert("RGB") #
    to_tensor = torchvision.transforms.ToTensor() #
    
    if args.defend_method != "cw":
        X = pil_to_input(init_image).to(device) #
        if dtype == torch.float16: # 如果是 CUDA 且非 CW，轉為 half
            X = X.half()
    else:
        X = to_tensor(init_image).to(device).unsqueeze(0) # CW 使用 float32

    # --- 3. 準備目標模型 ---
    model = None
    if args.target_model == "stable-diffusion":
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=dtype, # [CPU 支援] 使用動態 dtype
            safety_checker=None,
        ) #
    elif args.target_model == "instruct-pix2pix":
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            torch_dtype=dtype, # [CPU 支援] 使用動態 dtype
            safety_checker=None,
        ) #
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config) #
    else:
        raise ValueError(f"Invalid target_model '{args.target_model}'. Valid options are 'stable-diffusion' or 'instruct-pix2pix'.") #
    
    model.to(device) #

    # --- 4. 設定防禦 ---
    defend_fn = None
    if args.defend_method == "encoder":
        defend_fn = process_encoder_attack #
    elif args.defend_method == "vae":
        defend_fn = process_vae_attack #
    elif args.defend_method == "cw":
        defend_fn = process_cw_attack #
    elif args.defend_method == "facelock":
        defend_fn = process_facelock #
    else:
        raise ValueError(f"Invalid defend_method '{args.defend_method}'. Valid options are 'encoder', 'vae', 'cw', or 'facelock'.") #
    
    # --- 5. 執行防禦 ---
    print(f"[{__name__}] Processing defend method: {args.defend_method}...")
    X_adv = defend_fn(X, model, args, device) # 傳入 device
    print(f"[{__name__}] Defend complete.")

    # --- 6. 轉換回 PIL 圖像並儲存 ---
    to_pil = torchvision.transforms.ToPILImage() #
    if args.defend_method != "cw":
        X_adv = (X_adv / 2 + 0.5).clamp(0, 1) #
    
    # [CPU 支援] 確保 X_adv 是 float32 且在 CPU 上，以便轉換為 PIL
    protected_image = to_pil(X_adv[0].cpu().to(torch.float32)).convert("RGB")
    protected_image.save(args.output_path) #
    print(f"[{__name__}] Protected image saved to: {args.output_path}")

if __name__ == "__main__":
    parser = get_args_parser() #
    args = parser.parse_args() #

    main(args) #
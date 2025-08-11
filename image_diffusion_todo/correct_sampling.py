import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataset import tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDPMScheduler


def main():
    device = "cuda:0"
    
    # Load model
    print("Loading model...")
    ddpm = DiffusionModule(None, None)
    ddpm.load("../results/cfg_diffusion-ddpm-08-10-024322/last.ckpt")
    ddpm.eval()
    ddpm = ddpm.to(device)
    
    # Setup scheduler
    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
    ).to(device)
    
    # CFG scales to test
    cfg_scales = [0.0, 3.0, 7.5]
    
    # 각 클래스별로 1개씩만 생성
    print("\nGenerating samples...")
    
    for cfg_scale in cfg_scales:
        print(f"\n=== CFG {cfg_scale} ===")
        save_dir = Path(f"../correct_results/cfg_{cfg_scale}")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 고정 시드로 비교 가능하게
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # 각 클래스별로 1개씩
        class_labels = torch.tensor([0, 1, 2], dtype=torch.long).cuda()  # Cat, Dog, Wild
        
        samples = ddpm.sample(
            3,
            class_label=class_labels,
            guidance_scale=cfg_scale,
        )
        
        pil_images = tensor_to_pil_image(samples)
        
        for i, (img, class_name) in enumerate(zip(pil_images, ['cat', 'dog', 'wild'])):
            img.save(save_dir / f"{class_name}.png")
            print(f"  Saved {class_name}.png")
    
    # 비교 이미지 생성
    print("\n=== Creating comparison images ===")
    comp_dir = Path("../correct_results")
    
    for i, class_name in enumerate(['cat', 'dog', 'wild']):
        comp = Image.new('RGB', (64*3 + 20, 64), color='white')
        
        for j, cfg_scale in enumerate(cfg_scales):
            img_path = Path(f"../correct_results/cfg_{cfg_scale}/{class_name}.png")
            if img_path.exists():
                img = Image.open(img_path)
                comp.paste(img, (j*74, 0))
        
        comp_path = comp_dir / f"comparison_{class_name}.png"
        comp.save(comp_path)
        print(f"  Saved comparison_{class_name}.png")
    
    # 모든 비교 이미지를 하나로 합치기
    all_comp = Image.new('RGB', (64*3 + 20, 64*3 + 20), color='white')
    for i, class_name in enumerate(['cat', 'dog', 'wild']):
        comp_path = comp_dir / f"comparison_{class_name}.png"
        if comp_path.exists():
            comp = Image.open(comp_path)
            all_comp.paste(comp, (0, i*74))
    
    all_comp.save(comp_dir / "all_comparisons.png")
    print("\n  Saved all_comparisons.png")
    
    print("\nDone! Check correct_results/ folder")


if __name__ == "__main__":
    main()
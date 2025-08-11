import numpy as np
from PIL import Image
import os

# 비교 이미지 생성을 위한 스크립트
cfg_scales = ['0.0', '3.0', '7.5']
sample_indices = [100, 200, 300]

for idx in sample_indices:
    images = []
    for cfg in cfg_scales:
        img_path = f'../../results/report_images/comparison/cfg_{cfg}/sample_{idx}.png'
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
    
    if len(images) == 3:
        # 가로로 연결
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        
        new_im.save(f'comparison_{idx}.png')
        print(f'Created comparison_{idx}.png')

print('Comparison images created!')
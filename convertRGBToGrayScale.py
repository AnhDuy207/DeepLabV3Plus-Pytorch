from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# Đường dẫn tới ảnh gốc
image_path = '/home/duynguyen/AI/DeepLabV3Plus-Pytorch/seg_selfcar/03_00_197_seg.png'

target = Image.open(image_path).convert('L')
            
target = torch.tensor(np.array(target))

max_value = torch.max(target)

target = (target / max_value.item()) * 12
target = target.to(torch.uint8)

# Convert tensor to lbl
to_pil_transform = transforms.ToPILImage()
target = to_pil_transform(target)

# Lưu ảnh mới
target.save('/home/duynguyen/AI/DeepLabV3Plus-Pytorch/seg_selfcar/03_00_197_seg_GrayScale.png')
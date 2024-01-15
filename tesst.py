import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Đường dẫn đến file ảnh
image_path = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/testChangePixelValue/output.png"

# Đọc ảnh bằng Pillow
image_pil = Image.open(image_path)

# Chuyển đổi ảnh thành tensor PyTorch
transform = ToTensor()
image_tensor = transform(image_pil)

# In kích thước mới
print(image_tensor.size())
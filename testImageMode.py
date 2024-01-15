import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from utils import ext_transforms as et
from torchvision import transforms

def rgb_to_grayscale(rgb_image):
    # Chuyển đổi ảnh RGB sang Grayscale
    grayscale_image = rgb_image.convert("L")
    return grayscale_image

rgb_image_car = Image.open("/home/duynguyen/AI/DeepLabV3Plus-Pytorch/test_results/02_00_000.png")
grayscale_image_car = rgb_to_grayscale(rgb_image_car)

# rgb_image_city = Image.open("/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png")
# grayscale_image_city = Image.open("/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png")


print("tensor_rgb_image_car = ", torch.tensor(np.array(rgb_image_car)))
rgb_image_car = torch.tensor(np.array(rgb_image_car))
max_value = torch.max(rgb_image_car)
print("tensor_rgb_image_car => Max = ", max_value.item())

# print value of tensor before standardized
print("tensor_grayscale_image_car = ", torch.tensor(np.array(grayscale_image_car)))
# convert image to tensor
grayscale_image_car = torch.tensor(np.array(grayscale_image_car))

print("tensor_grayscale_image_car = ", torch.tensor(np.array(grayscale_image_car)))
max_value = torch.max(grayscale_image_car)
print("tensor_rgb_image_car => Max = ", max_value.item())



# [0,255] -> [0,13]
grayscale_image_car = (grayscale_image_car / max_value.item()) * 13
# sign to Int
grayscale_image_car = grayscale_image_car.to(torch.int)
# print value of tensor after standardized
print("tensor_grayscale_image_car = ", torch.tensor(np.array(grayscale_image_car)))
# Find max value in tensor
max_value = torch.max(grayscale_image_car)
print("tensor_grayscale_image_car => Max = ", max_value.item())

to_pil_transform = transforms.ToPILImage()
target = to_pil_transform(grayscale_image_car)



# print("tensor_rgb_image_city = ", torch.tensor(np.array(rgb_image_city)))
# rgb_image_city = torch.tensor(np.array(rgb_image_city))
# max_value = torch.max(rgb_image_city)
# print("tensor_rgb_image_city => Max = ", max_value.item())

# print("tensor_grayscale_image_city = ", torch.tensor(np.array(grayscale_image_city)))
# grayscale_image_city = torch.tensor(np.array(grayscale_image_city))
# max_value = torch.max(grayscale_image_city)
# print("tensor_grayscale_image_city => Max = ", max_value.item())

# Lưu ảnh Grayscale hoặc hiển thị
target.save("grayscale_image.png")
target.show()


# import cv2
# import numpy as np

# # Đọc ảnh grayscale
# image = cv2.imread('/home/duynguyen/AI/DeepLabV3Plus-Pytorch/test_results/02_00_000.png')

# # Chuyển đổi ảnh sang grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Áp dụng phân đoạn để trích xuất đối tượng
# # Ví dụ: Sử dụng phương pháp phân ngưỡng đơn giản
# _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# # Chuyển đổi các đối tượng thành ảnh grayscale
# segmented_image = cv2.bitwise_and(gray_image, gray_image, mask=thresholded_image)

# # segmented_image = cv2.medianBlur(segmented_image, 3)


# # convert image to tensor
# segmented_image = torch.tensor(np.array(segmented_image))

# print("tensor_segmented_image = ", torch.tensor(np.array(segmented_image)))
# max_value = torch.max(segmented_image)
# print("segmented_image => Max = ", max_value.item())



# # [0,255] -> [0,13]
# # segmented_image = (segmented_image / max_value.item()) * 13
# # sign to Int
# segmented_image = segmented_image.to(torch.int)
# # print value of tensor after standardized
# print("tensor_grayscale_image_car = ", torch.tensor(np.array(segmented_image)))
# # Find max value in tensor
# max_value = torch.max(segmented_image)
# print("tensor_grayscale_image_car => Max = ", max_value.item())

# to_pil_transform = transforms.ToPILImage()
# target = to_pil_transform(segmented_image)



# # print("tensor_rgb_image_city = ", torch.tensor(np.array(rgb_image_city)))
# # rgb_image_city = torch.tensor(np.array(rgb_image_city))
# # max_value = torch.max(rgb_image_city)
# # print("tensor_rgb_image_city => Max = ", max_value.item())

# # print("tensor_grayscale_image_city = ", torch.tensor(np.array(grayscale_image_city)))
# # grayscale_image_city = torch.tensor(np.array(grayscale_image_city))
# # max_value = torch.max(grayscale_image_city)
# # print("tensor_grayscale_image_city => Max = ", max_value.item())

# # Lưu ảnh Grayscale hoặc hiển thị
# target.save("grayscale_image.png")
# target.show()
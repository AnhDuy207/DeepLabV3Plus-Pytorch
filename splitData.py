import os
import shutil
import random
from tqdm import tqdm


def split_image_folder(input_folder_img, output_folder1_img, output_folder2_img, split_ratio):
    file_list = os.listdir(input_folder_img)
    file_list = sorted(file_list)
    image_list = [file for file in file_list if file.endswith((".png"))]

    num_images = len(image_list)
    num_images_folder1 = int(num_images * split_ratio)

    random_numbers = random.sample(range(0, num_images), num_images_folder1)
    random_numbers = sorted(random_numbers)
    
    # Tao thu muc dau ra neu chua ton tai
    os.makedirs(output_folder1_img, exist_ok=True)
    os.makedirs(output_folder2_img, exist_ok=True)
    
    # Chia anh thanh hai thu muc
    for i, image in tqdm(enumerate(image_list)):
        isTrain = 1
        for j in random_numbers:
            if i==j: 
                image_path = os.path.join(input_folder_img, image_list[j])
                output_path = os.path.join(output_folder1_img, image)
                shutil.copy(image_path, output_path)
                isTrain = 0
        if isTrain == 1:
            image_path = os.path.join(input_folder_img, image_list[i])
            output_path = os.path.join(output_folder2_img, image)
            shutil.copy(image_path, output_path)   

    print("Chia muc Image thanh hai thu muc thanh cong!")
    return random_numbers

def split_ann_folder(input_folder_img, output_folder1_img, output_folder2_img, split_ratio, random_numbers):
    file_list = os.listdir(input_folder_img)
    file_list = sorted(file_list)
    image_list = [file for file in file_list if file.endswith((".json"))]

    num_images = len(image_list)
    
    # Tao thu muc dau ra neu chua ton tai
    os.makedirs(output_folder1_img, exist_ok=True)
    os.makedirs(output_folder2_img, exist_ok=True)
    
    # Chia anh thanh hai thu muc
    for i, image in tqdm(enumerate(image_list)):
        isTrain = 1
        for j in random_numbers:
            if i==j: 
                image_path = os.path.join(input_folder_img, image_list[j])
                output_path = os.path.join(output_folder1_img, image)
                shutil.copy(image_path, output_path)
                isTrain = 0
        if isTrain == 1:
            image_path = os.path.join(input_folder_img, image_list[i])
            output_path = os.path.join(output_folder2_img, image)
            shutil.copy(image_path, output_path)   

    print("Chia muc Ann thanh hai thu muc thanh cong!")
    return random_numbers

# Su dug vi du
# input_folder_img = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/self-driving-cars-DatasetNinja/ds/img"
output_folder1_img = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/img/test"
output_folder2_img = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/img/train"

# input_folder_ann = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/self-driving-cars-DatasetNinja/ds/ann"
output_folder1_ann = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/ann/test"
output_folder2_ann = "/home/duynguyen/AI/DeepLabV3Plus-Pytorch/datasets/data/selfcar/ann/train"
split_ratio = 0.2  #Chia ti le

# Split Img to test and train folder
random_numbers = split_image_folder(input_folder_img, output_folder1_img, output_folder2_img, split_ratio)
# Split Ann to test and train folder
split_ann_folder(input_folder_ann, output_folder1_ann, output_folder2_ann, split_ratio, random_numbers)
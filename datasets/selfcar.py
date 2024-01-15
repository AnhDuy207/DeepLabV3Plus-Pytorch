import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2

# import supervisely as sly
from tqdm import tqdm
from torchvision import transforms

class Selfcar(data.Dataset):
    SelfcarClass = namedtuple('SelfcarClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        # SelfcarClass('road',                    0, 0, 'flat', 0, False, False, (0, 0, 0)), #den
        # SelfcarClass('other',                   1, 255, 'flat', 0, False, False, (255, 255, 255)),
        # SelfcarClass('other',                   2, 255, 'void', 1, False, True, (255, 255, 255)),
        # SelfcarClass('other',                   3, 255, 'human', 2, True, False, (255, 255, 255)),
        # SelfcarClass('car',                     4, 1, 'vehicle', 3, True, False, (255, 128, 0)), #cam
        # SelfcarClass('other',                   5, 255, 'construction', 4, False, False, (255, 255, 255)),
        # SelfcarClass('building',                6, 2, 'construction', 4, False, False, (255, 255, 0)), #vang
        # SelfcarClass('other',                   7, 255, 'construction', 4, False, False, (255, 255, 255)),
        # SelfcarClass('other',                   8, 255, 'nature', 5, False, False, (255, 255, 255)),
        # SelfcarClass('other',                   9, 255, 'object', 6, False, False, (255, 255, 255)),
        # SelfcarClass('other',                   10, 255, 'object', 6, False, False, (255, 255, 255)),
        # SelfcarClass('tree',                    11, 3, 'nature', 5, False, False, (0, 255, 0)), #xanh
        # SelfcarClass('sky',                     12, 4, 'sky', 7, False, False, (255, 153, 204)), #hong
        SelfcarClass('road',                    0, 0, 'flat', 0, False, False, (0, 0, 0)), #den
        SelfcarClass('other',                   1, 1, 'flat', 0, False, False, (255, 255, 255)),
        SelfcarClass('other',                   2, 2, 'void', 1, False, True, (255, 255, 255)),
        SelfcarClass('other',                   3, 3, 'human', 2, True, False, (255, 255, 255)),
        SelfcarClass('car',                     4, 4, 'vehicle', 3, True, False, (255, 128, 0)), #cam
        SelfcarClass('other',                   5, 5, 'construction', 4, False, False, (255, 255, 255)),
        SelfcarClass('building',                6, 6, 'construction', 4, False, False, (255, 255, 0)), #vang
        SelfcarClass('other',                   7, 7, 'construction', 4, False, False, (255, 255, 255)),
        SelfcarClass('other',                   8, 8, 'nature', 5, False, False, (255, 255, 255)),
        SelfcarClass('other',                   9, 9, 'object', 6, False, False, (255, 255, 255)),
        SelfcarClass('other',                   10, 10, 'object', 6, False, False, (255, 255, 255)),
        SelfcarClass('tree',                    11, 11, 'nature', 5, False, False, (0, 255, 0)), #xanh
        SelfcarClass('sky',                     12, 12, 'sky', 7, False, False, (255, 153, 204)), #hong
        SelfcarClass('other',                   -1, 255, 'vehicle', 7, False, True, (255, 255, 255)),
    ]
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    def __init__(self, root, split = 'train', transform = None, isSelfCar = 1):
        self.isSelfCar = isSelfCar
        if self.isSelfCar == 1:
            self.root = os.path.expanduser(root)
            
            self.images_dir = os.path.join(self.root, 'img', split)
            self.targets_dir = os.path.join(self.root, 'segment', split)
            self.transform = transform

            self.split = split
            self.images = []
            self.targets = []
            
            if split not in ['train', 'test', 'val']:
                raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                                ' or split="val"')
            
            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                ' specified "split" and "mode" are inside the "root" directory')
            
            for image in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, image))
                
            for target in os.listdir(self.targets_dir):
                self.targets.append(os.path.join(self.targets_dir, target))
                
        elif self.isSelfCar == 0:
            self.root = os.path.expanduser('./datasets/data/cityscapes_Custom/')
            self.mode = 'gtFine'
            self.target_type = 'semantic'
            self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

            self.targets_dir = os.path.join(self.root, self.mode, split)
            self.transform = transform

            self.split = split
            self.images = []
            self.targets = []

            if split not in ['train', 'test', 'val']:
                raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                                ' or split="val"')

            if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                ' specified "split" and "mode" are inside the "root" directory')
            
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)

                for file_name in os.listdir(img_dir):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                self._get_target_suffix(self.mode, self.target_type))
                    self.targets.append(os.path.join(target_dir, target_name))
    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]
        

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 13
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        if self.isSelfCar == 1: 
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.targets[index]).convert('L')
            
            target = torch.tensor(np.array(target))
            
            max_value = torch.max(target)
            
            target = (target / max_value.item()) * 12
            target = target.to(torch.uint8)

            # Convert tensor to lbl
            to_pil_transform = transforms.ToPILImage()
            target = to_pil_transform(target)
            
            if self.transform:
                image, target = self.transform(image, target)   
            target = self.encode_target(target)
            return image, target
        
        elif self.isSelfCar == 0:  
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.targets[index])
            
            # # Chuyển đổi ảnh PIL thành mảng numpy
            # target = np.array(target)

            # # Lấy kích thước ảnh
            # height, width = target.shape

            # # In giá trị từng pixel
            # for y in range(height):
            #     for x in range(width):
            #         #road
            #         if target[y, x] == 7:
            #             target[y, x] = 0
            #         #building
            #         elif target[y, x] == 11:
            #             target[y, x] = 6
            #         #tree
            #         elif target[y, x] == 21:
            #             target[y, x] = 11
            #         #sky
            #         elif target[y, x] == 23:
            #             target[y, x] = 12
            #         #car
            #         elif target[y, x] == 26:
            #             target[y, x] = 4   
            #         else:
            #             target[y, x] = 1

            # # Tạo đối tượng Image từ mảng numpy đã chuyển đổi
            # target = Image.fromarray(target)
            
            if self.transform:
                image, target = self.transform(image, target)
            target = self.encode_target(target)
            return image, target
        
    def __len__(self):
        return len(self.images)
    
    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    
    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
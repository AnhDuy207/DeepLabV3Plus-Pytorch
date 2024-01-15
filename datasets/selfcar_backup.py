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
        SelfcarClass('sky',                     0, 10, 'sky', 5, False, False, (70, 130, 180)),
        SelfcarClass('building',                1, 2, 'construction', 2, False, False, (70, 70, 70)),
        SelfcarClass('fence',                   2, 4, 'construction', 2, False, False, (190, 153, 153)),
        SelfcarClass('street infrastructure',   3, 9, 'nature', 4, False, False, (152, 251, 152)),
        SelfcarClass('pole',                    4, 5, 'object', 3, False, False, (153, 153, 153)),
        SelfcarClass('road markings',           5, 255, 'void', 0, False, True, (0, 0, 0)),
        SelfcarClass('road',                    6, 0, 'flat', 1, False, False, (128, 64, 128)),
        SelfcarClass('sidewalk',                7, 1, 'flat', 1, False, False, (244, 35, 232)),
        SelfcarClass('tree',                    8, 8, 'nature', 4, False, False, (107, 142, 35)),
        SelfcarClass('car',                     9, 12, 'vehicle', 7, True, False, (0, 0, 142)),
        SelfcarClass('wall',                    10, 3, 'construction', 2, False, False, (102, 102, 156)),
        SelfcarClass('traffic',                 11, 6, 'object', 3, False, False, (250, 170, 30)),
        SelfcarClass('pedestrian',              12, 11, 'human', 6, True, False, (220, 20, 60)),      
    ]
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    def __init__(self, root, split = 'train', transform = None, isSelfCar = 1):
        self.isSelfCar = isSelfCar
        if self.isSelfCar == 1:
            print("Nhay zo ham SelfCar -> init")
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
            print("Nhay zo ham Cityscapes -> init")
            self.root = os.path.expanduser('./datasets/data/cityscapes/')
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
        return cls.train_id_to_color[target-1]

    def __getitem__(self, index):
        if self.isSelfCar == 1:
            print("Nhay zo ham SelfCar -> __getitem__")   
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
            print("Nhay zo ham Cityscapes -> __getitem__")   
            image = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.targets[index])
            
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
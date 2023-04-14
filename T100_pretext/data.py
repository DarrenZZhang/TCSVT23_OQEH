from PIL import Image, ImageFilter
from numpy.core.fromnumeric import resize
import cv2
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import transforms


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self):
        self.base_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.augm_transform = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply(
                                                      [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlur(3), # add
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                                       [0.229, 0.224, 0.225])
                                                  ])

    def __call__(self, raw_img):
        view_1 = self.base_transform(raw_img)
        view_2 = self.augm_transform(raw_img)
        return [view_1, view_2]

class ImgDataset(torch.utils.data.Dataset):
    """Some Information about ImgDataset"""
    def __init__(self, data=None, targets=None, transform=None, target_transform=None, img_root=None):
        super(ImgDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.img_root = img_root

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.img_root:
            img = Image.open(os.path.join(self.img_root, img)).convert('RGB')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)

class DATASET_step1:
    def __init__(self,
                 root,
                 img_root,
                 batch_size=128,
                 num_workers=4):
        '''
            root: root of image path file
            img_root: root of image file
        '''
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Fetch data
        with open(os.path.join(root, 'train_file_list.txt'), 'r') as image_file:
            train_data = np.array([i.strip() for i in image_file])
        train_targets = np.loadtxt(os.path.join(root, 'train_label.txt'), dtype=np.int8)

        # two views
        self._train_dataset = ImgDataset(data=train_data,
                                         targets=train_targets,
                                         transform=TwoCropsTransform(),
                                         # transform=test_transforms,
                                         img_root=img_root)
        # one view
        self._train_dataset_eval = ImgDataset(data=train_data,
                                         targets=train_targets,
                                         transform=test_transforms,
                                         img_root=img_root)

        # Setup data loaders
        # shuffle
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True, # 每个epochs乱序
                                                         num_workers=num_workers)
        # no shuffle
        self._train_loader_eval = torch.utils.data.DataLoader(dataset=self._train_dataset_eval,
                                                              batch_size=batch_size,
                                                              shuffle=False, # 每个epochs不乱�?
                                                              num_workers=num_workers)
    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataset_eval(self):
        return self._train_dataset_eval

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def train_loader_eval(self):
        return self._train_loader_eval
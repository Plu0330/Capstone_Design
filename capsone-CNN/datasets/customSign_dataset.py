import os
from datasets.base_dataset import get_transform
from datasets.base_dataset import BaseDataset
import torch
import numpy as np
from imageio import imread
from PIL import Image
import torchvision.transforms as transforms
import math
import natsort
import json


class SignDataset(BaseDataset):
    """Represents a 2D segmentation dataset.
    
    Input params:
        configuration: Configuration dictionary.
    """
    # def __init__(self, configuration):
    #     super().__init__(configuration)
    #     self.img_dir = configuration['dataset_path']
    #     self.list_dir = os.listdir(self.img_dir)
    #     #self.list_dir = natsort.natsorted(self.list_dir)
    #     self.label_list = {}
    #     for num, label in enumerate(self.list_dir):
    #         label_name = str(label)
    #         self.label_list[label_name] = num
            
    #     self.img_paths = []
    #     self.img_transforms = transform_image()
    #     for classes in self.list_dir:
    #         for path, _, files in os.walk(os.path.join(self.img_dir, classes)):
    #             for file in files:
    #                 file_path = os.path.join(path, file)
    #                 self.img_paths += [[file_path, self.label_list[classes]]]
    #     assert len(self.img_paths) != 0
     
        
    # 도로안내표지 표지판 순서대로 data저장
    def __init__(self, configuration):
        super().__init__(configuration)
        self.img_dir = configuration['dataset_path']
        self.list_dir = os.listdir(self.img_dir)
        
        self.label_list = {}
        for num, label in enumerate(self.list_dir):
            label_name = str(label)
            self.label_list[label_name] = num
            
        self.img_paths = []
        self.img_transforms = transform_image()
        damaged_annotations = json.load(open(r'D:\ms\capstone\dataset\custom\damaged_annotation.json', encoding='utf8'))
        for road_sign in damaged_annotations:
            for region in damaged_annotations[road_sign]:
                region_path = os.path.join(self.img_dir,region['bbox_text'])
                file_name = road_sign[:-4] + '_{:04}.jpg'.format(int(region['id']))
                file_path = os.path.join(region_path,file_name)
                self.img_paths += [[file_path, self.label_list[region['bbox_text']]]]
        assert len(self.img_paths) != 0
        
        #한 표지판에 몇개의 region이 있는지 = road_num
        self.road_num = []
        for d in damaged_annotations:
            self.road_num.append(len(damaged_annotations[d]))
        #print(self.road_num)
        with open(r'C:\Users\ms9804\Desktop\capstone\5.capstoneTestVer1\for_merge\road_num.json','w') as f:
            json.dump(self.road_num, f, indent=4)
        

    def __getitem__(self, index):
        # get source image as x
        # get labels as y
        x = Image.open(self.img_paths[index][0])
        x = x.convert('RGB')
        x = self.img_transforms(x)
        #x = np.asarray(x)
        #y = self.to_categorical(int(self.img_paths[index][1]))
        y = int(self.img_paths[index][1])
        return (x, y)

    def __len__(self):
        # return the size of the dataset
        return len(self.img_paths)

    def load_image(self, paths):
        #X1 = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        X = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.n_channels))
        y = np.empty((self.batch_size, self.num_classes))
        for path, classes in paths:
            img = imread(path)
            X[i,] = img
            y[i,] = to_categorical(classes)
        return (X, y)

    def to_categorical(self, y):
        return np.eye(self.num_classes, dtype='longlong')[y]


class ToSpaceBGR(object):
        def __init__(self, is_bgr):
            self.is_bgr = is_bgr

        def __call__(self, tensor):
            if self.is_bgr:
                new_tensor = tensor.clone()
                new_tensor[0] = tensor[2]
                new_tensor[2] = tensor[0]
                tensor = new_tensor
            return tensor

class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor
    
    
class transform_image(object):
    def __init__(self, input_size=[3, 224, 224], input_space='RGB', input_range=[0, 1], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], scale=0.875, random_crop=False, random_hflip=False, random_vflip=False, preserve_aspect_ratio=True,
                 random_rotation = False, colorjitter = False, random_resized_crop=False, gaussian_blur=False,
                 random_erasing=False):
        input_size = input_size
        input_space = input_space
        input_range = input_range
        mean = mean
        std = std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        scale = scale
        random_crop = random_crop
        random_hflip = random_hflip
        random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(input_size)/scale))))
        else:
            height = int(input_size[1] / scale)
            width = int(input_size[2] / scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        if random_rotation:
            tfs.append(transforms.RandomRotation(degrees=45))
            print('aug = random_rotation')

        if colorjitter:
            #tfs.append(transforms.RandomApply(transforms = [transforms.ColorJitter(brightness=0.9, hue=0.3)], p=0.5))
            tfs.append(transforms.ColorJitter(brightness=0.5))
            print('aug = colorjitter')

        if random_resized_crop:
            tfs.append(transforms.RandomResizedCrop(max(input_size)))
            print('aug = random_resized_crop')

        if gaussian_blur:
            tfs.append(transforms.RandomApply(transforms= [transforms.GaussianBlur(3,5)], p=0.5))
            #tfs.append(transforms.GaussianBlur(3,5))
            print('aug = gausian0.5')

        if random_rotation:
            tfs.append(transforms.RandomApply(transforms = [transforms.RandomRotation(degrees=45)],p=0.5))
            #tfs.append(transforms.RandomRotation(degrees=45))

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(input_space=='BGR'))
        tfs.append(ToRange255(max(input_range)==255))
        tfs.append(transforms.Normalize(mean=mean, std=std))
        
        if random_erasing:
            tfs.append(transforms.RandomErasing(p=0.5))
            print('aug = random_erasing')

        self.img_tf = transforms.Compose(tfs)
        #return self.img_tf

    def __call__(self, img):
        tensor = self.img_tf(img)
        return tensor

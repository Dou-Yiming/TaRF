import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import json


class YCBBase(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 main_modality='image',
                 seg_mask=False,
                 mask_size=None,
                 rotation=0,
                 set_type='train'
                 ):
        assert main_modality in ['image', 'touch']
        self.data_paths = json_file
        self.data_root = data_root
        self.set_type = set_type

        self.object_list = [
            '004_sugar_box',
            '005_tomato_soup_can',
            '006_mustard_bottle',
            '021_bleach_cleanser',
            '025_mug',
            '035_power_drill',
            '037_scissors',
            '042_adjustable_wrench',
            '048_hammer',
            '055_baseball'
        ]

        # load candidates
        self.cand = []
        with open(json_file) as f:
            # [[obj, contact]]
            self.cand = json.load(f)[self.set_type]
        if self.set_type == 'val':
            self.cand = self.cand[::50]
        self._length = len(self.cand)
        self.data = {
            "image_path_": [os.path.join(self.data_root, 'vision', self.cand[i][0], self.cand[i][1]+'.jpg') for i in range(len(self.cand))],
            "touch_path_": [os.path.join(self.data_root, 'touch', self.cand[i][0], self.cand[i][1]+'.jpg') for i in range(len(self.cand))],
        }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality
        
        # rotation
        self.rotation = rotation

        

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)
        
        # load visual image
        if self.main_modality == 'image':
            image = Image.open(example["image_path_"])
        elif self.main_modality == 'touch':
            image = Image.open(example["touch_path_"])
        else:
            print('No matched modality')
            exit()
        if not image.mode == "RGB":
            image = image.convert("RGB")
    

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        # load aux image
        if self.main_modality == 'image':
            image = Image.open(example["touch_path_"])
        elif self.main_modality == 'touch':
            image = Image.open(example["image_path_"])
        else:
            print('No matched modality')
            exit()
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["aux"] = (image / 127.5 - 1.0).astype(np.float32)
       
        return example
    
    def load_test_img(self, path):
        image = Image.open(path).convert("RGB")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
        

        # image = self.flip(image)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.


class TouchTrain(YCBBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/ycb_slide/split.json", data_root="/nfs/turbo/coe-ahowens/datasets/ycb_slide/", flip_p=flip_p, set_type='train', **kwargs)

class TouchValidation(YCBBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/ycb_slide/split.json", data_root="/nfs/turbo/coe-ahowens/datasets/ycb_slide/", flip_p=flip_p, set_type='val', **kwargs)

class TouchTest(YCBBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/ycb_slide/split.json", data_root="/nfs/turbo/coe-ahowens/datasets/ycb_slide/", flip_p=flip_p, set_type='val', **kwargs)
    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)
        
        # load visual image
        if self.main_modality == 'image':
            example["image"] = self.load_test_img(example["image_path_"])
        elif self.main_modality == 'touch':
            example["image"] = self.load_test_img(example["touch_path_"])
        else:
            print('No matched modality')
            exit()
        
        # load aux image
        if self.main_modality == 'image':
            example["aux"] = self.load_test_img(example["touch_path_"])
        elif self.main_modality == 'touch':
            example["aux"] = self.load_test_img(example["image_path_"])
        else:
            print('No matched modality')
            exit()

        # rotate aux
        if self.rotation != 0:
            example["aux"] = transforms.functional.rotate(example["aux"], angle=self.rotation)

        return example

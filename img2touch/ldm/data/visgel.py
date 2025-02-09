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


class VisGel(Dataset):
    def __init__(self,
                 data_root,
                 txt_file="",
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 main_modality='image',
                 mode='train',
                 rotation=0
                 ):
        assert main_modality in ['image', 'touch']
        assert mode in ['train', 'val']
        if main_modality == 'image':
            if mode == 'train':
                self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/train_touch2vision.txt"
                # self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/visgel_touching.txt"
            else:
                self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/eval_touch2vision.txt"
        elif main_modality == 'touch':
            if mode == 'train':
                self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/train_vision2touch.txt"
                # self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/train_vision2touch_full.txt"
                # self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/eval_vision2touch.txt"
                self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/visgel_touching.txt"
            else:
                self.data_paths = "/nfs/turbo/coe-ahowens/fredyang/VisGel/data_lst/eval_vision2touch.txt"
        else:
            print('modality unimplemented')
            exit()

        # self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            # self.image_paths = f.read().splitlines()
            self.raw_data = f.read().splitlines()
        
        # if len(self.raw_data) > 200000:
        #     self.raw_data = self.raw_data[:200000]
        
        self._length = len(self.raw_data)
        self.data = {
            "relative_file_path_": [self.raw_data[i] for i in range(len(self.raw_data))]
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
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

        ref_src, ref_des, src, des, src_pre_0, src_pre_1, src_nxt_0, src_nxt_1 = example["relative_file_path_"].strip().split(" ")
        
        # load main image
        main_path = os.path.join(self.data_root, des)
        image = Image.open(main_path)
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
        main_path = os.path.join(self.data_root, src)
        image = Image.open(main_path)
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
        tmp_aux = (image / 127.5 - 1.0).astype(np.float32)

        # load ref image
        main_path = os.path.join(self.data_root, ref_des)
        image = Image.open(main_path)
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
        ref = (image / 127.5 - 1.0).astype(np.float32)
        # example["ref"] = np.concatenate([ref, tmp_aux], axis=2)
        example["ref"] = np.concatenate([ref, tmp_aux], axis=2)
        # print(example["ref"].shape)
       
       
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


class VisGelTrain(VisGel):
    def __init__(self, **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/fredyang/VisGel", mode='train', **kwargs)


class VisGelValidation(VisGel):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/fredyang/VisGel", mode='val', flip_p=flip_p, **kwargs)

class VisGelTest(VisGel):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/fredyang/VisGel", mode='val', flip_p=flip_p, **kwargs)
    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)

        ref_src, ref_des, src, des, src_pre_0, src_pre_1, src_nxt_0, src_nxt_1 = example["relative_file_path_"].strip().split(" ")

        # load main image
        main_path = os.path.join(self.data_root, des)
        example["image"] = self.load_test_img(main_path)

        # load aux image
        main_path = os.path.join(self.data_root, src)
        example["aux"] = self.load_test_img(main_path)
        tmp_aux = example["aux"]

        # rotate aux
        if self.rotation != 0:
            tmp_aux = transforms.functional.rotate(tmp_aux, angle=self.rotation)

        #load ref image
        main_path = os.path.join(self.data_root, ref_des)
        ref = self.load_test_img(main_path)
        example["ref"] = torch.cat([ref, tmp_aux], dim=1)


        return example

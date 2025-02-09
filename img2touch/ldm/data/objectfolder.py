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


class TouchBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 main_modality='image',
                 seg_mask=False,
                 mask_size=None,
                 rotation=0
                 ):
        assert main_modality in ['image', 'touch']
        self.data_root = data_root
        self.image_paths = os.listdir(os.path.join(self.data_root, 'vision'))
        
        # self.sub_folder, self.image_paths, self.image_label = [], [], []
        # for raw_path in self.raw_data:
        #     self.image_paths.append(raw_path)

        
        self._length = len(self.image_paths)
        self.data = {
            "image_path_": [os.path.join(self.data_root, 'vision', self.image_paths[i]) for i in range(len(self.image_paths))],
            "touch_path_": [os.path.join(self.data_root, 'touch', self.image_paths[i]) for i in range(len(self.image_paths))],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality

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


class TouchTrain(TouchBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/ymdou/share/objectfolder_50/train/", **kwargs)


class TouchValidation(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/ymdou/share/objectfolder_50/val/", flip_p=flip_p, **kwargs)

class TouchTest(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_root="/nfs/turbo/coe-ahowens/ymdou/share/objectfolder_50/test/", flip_p=flip_p, **kwargs)
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

        return example


class TDIS(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_clip_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
    
    def __getitem__(self, i):
        # set up A, B
        A_index = i
        B_index = random.randint(0,self._length)
        # B_index = A_index
        example = dict((k, self.data[k][A_index]) for k in self.data)
        example2 = dict((k, self.data[k][B_index]) for k in self.data)
        
        # load visual image
        example["imageA"] = self.load_test_img(example["image_path_"])
        example["gelsightA"] = self.load_test_img(example["touch_path_"])
        example["imageB"] = self.load_test_img(example2["image_path_"])
        example["gelsightB"] = self.load_test_img(example2["touch_path_"])


        return example
    
class TDIS_control(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_tdis.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
    
    def __getitem__(self, i):
        with open('/home/fredyang/fredyang/stable-diffusion/data/touch_and_go/tdis_test_control_index.txt', "r") as f:
            B_list = f.read().splitlines()

        # set up A, B
        A_index = i
        B_index = int(B_list[i])
        example = dict((k, self.data[k][A_index]) for k in self.data)
        example2 = dict((k, self.data[k][B_index]) for k in self.data)
        
        # load visual image
        example["imageA"] = self.load_test_img(example["image_path_"])
        example["gelsightA"] = self.load_test_img(example["touch_path_"])
        example["imageB"] = self.load_test_img(example2["image_path_"])
        example["gelsightB"] = self.load_test_img(example2["touch_path_"])


        return example
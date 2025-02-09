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
#from image_bind import data
import pandas as pd


class TouchBase(Dataset):
    def __init__(self,
                 txt_file,
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
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            # self.image_paths = f.read().splitlines()
            self.raw_data = f.read().splitlines()
        
        self.sub_folder, self.image_paths, self.image_label = [], [], []
        for raw in self.raw_data:
            raw_path, raw_label = raw.strip().split(',')
            self.sub_folder.append(raw_path[:15])
            self.image_paths.append(raw_path[16:])
            self.image_label.append(raw_label)

        # print(self.sub_folder[0], self.image_paths[0], self.image_label[0])
        # exit()
        # self.image_paths = self.image_paths[:100]
        
        self._length = len(self.image_paths)
        self.data = {
            "relative_file_path_": [os.path.join(self.sub_folder[i], self.image_paths[i]) for i in range(len(self.image_paths))],
            "image_path_": [os.path.join(self.data_root, self.sub_folder[i], 'video_frame', self.image_paths[i]) for i in range(len(self.image_paths))],
            "touch_path_": [os.path.join(self.data_root, self.sub_folder[i], 'gelsight_frame', self.image_paths[i]) for i in range(len(self.image_paths))],
            "class_label": [int(self.image_label[i]) for i in range(len(self.image_paths))],
        }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality

        self.seg_mask = seg_mask
        self.seg_root = '/nfs/turbo/coe-ahowens/datasets/touch_and_go/segmentation'
        
        self.torch_resize = None
        if self.size is not None:
            self.torch_resize = transforms.Resize([self.size, self.size])

        self.mask_size = mask_size
        
        if self.mask_size is not None:
            self.mask_resize = transforms.Resize([self.mask_size, self.mask_size])
            if self.size is not None:
                self.max_pool = nn.MaxPool2d(int(self.size / self.mask_size))
        
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

        # # 实验：mask手抹黑
        # seg_path = os.path.join(self.seg_root, example['relative_file_path_'][:15] + '_' + example['relative_file_path_'][16:-4] + '.npy')
        # seg_mask = np.load(seg_path)
        # if seg_mask.shape[0] > 1:
        #     seg_mask = np.sum(seg_mask, axis=0, keepdims=True)
        # assert seg_mask.shape[0] == 1

        #  # crop accordingly
        # crop = min(seg_mask.shape[0], seg_mask.shape[1])
        # h, w, = seg_mask.shape[0], seg_mask.shape[1]
        # seg_mask = seg_mask[(h - crop) // 2:(h + crop) // 2,
        #     (w - crop) // 2:(w + crop) // 2]
        # seg_mask = torch.from_numpy(seg_mask)

        # if self.size is not None:
        #     seg_mask = seg_mask.resize((self.size, self.size), resample=self.interpolation)

        # # print(seg_mask.shape)
        # seg_mask = np.swapaxes(seg_mask, 0, 2)
        # seg_mask = np.swapaxes(seg_mask, 0, 1)
        # seg_mask = np.repeat(seg_mask, 3, 2)
        # # print(img.shape,seg_mask.shape)
        # image[seg_mask] = 0

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

        # load segmentation mask
        # print(example['relative_file_path_'])
        # print(example['relative_file_path_'][:15], example['relative_file_path_'][16:])

        if self.seg_mask == True:
            seg_path = os.path.join(self.seg_root, example['relative_file_path_'][:15] + '_' + example['relative_file_path_'][16:-4] + '.npy')
            seg_mask = np.load(seg_path)
            if seg_mask.shape[0] > 1:
                seg_mask = np.sum(seg_mask, axis=0, keepdims=True)
            assert seg_mask.shape[0] == 1

            # swap axis
            seg_mask = np.swapaxes(seg_mask, 0, 2)
            seg_mask = np.swapaxes(seg_mask, 0, 1)

            # crop accordingly
            crop = min(seg_mask.shape[0], seg_mask.shape[1])
            h, w, = seg_mask.shape[0], seg_mask.shape[1]
            seg_mask = seg_mask[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
            seg_mask = torch.from_numpy(seg_mask)
            
            # print(seg_mask.shape, seg_path)
            # 暂时去掉空矩阵
            # if seg_mask.shape[0] == 0:
            #     seg_mask = torch.zeros((1,seg_mask.shape[1],seg_mask.shape[2]) , dtype=torch.bool)
                # seg_mask = torch.ones((1,seg_mask.shape[1],seg_mask.shape[2]) , dtype=torch.bool)
            # print(torch.sum(seg_mask).item())
            
            if self.size is not None:
                # print(seg_mask.shape)
                seg_mask = torch.permute(seg_mask, (2, 0, 1))
                seg_mask = self.torch_resize(seg_mask)
                # seg_mask = seg_mask.float()
                # print(seg_mask.shape, int(seg_mask.shape[1] / self.size))
                # seg_mask = F.MaxPool2d(seg_mask, int(seg_mask.shape[1] / self.size))
                # seg_mask = seg_mask.int()
                # seg_mask = seg_mask.bool()
            seg_mask_dup = seg_mask
            if self.mask_size is not None:
                # seg_mask = self.mask_resize(seg_mask)
                seg_mask = seg_mask.float()
                seg_mask = self.max_pool(seg_mask)
                seg_mask = seg_mask.int()
                seg_mask = seg_mask.bool()
            
            
            # 实验：mask吧手抹黑
            # seg_mask_numpy = seg_mask_dup.numpy()
            # seg_mask_numpy = np.swapaxes(seg_mask_numpy, 0, 2)
            # seg_mask_numpy = np.swapaxes(seg_mask_numpy, 0, 1)
            # # print(example["image"].shape, seg_mask_numpy.shape)

            # # print(example["image"].shape, seg_mask_numpy.shape)
            # seg_mask_numpy = np.repeat(seg_mask_numpy, 3, 2)
            # # print(seg_mask_numpy.shape)
            # example["image"][seg_mask_numpy] = 1
            # # print(example["image"].shape)
            # # exit()
             
            example['seg_mask'] = seg_mask[0,:,:]
            # print(torch.sum(example['seg_mask']).item())
       
       
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


# class TouchTrain(TouchBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/touch_and_go/train_clip.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", **kwargs)


# class TouchValidation(TouchBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/touch_and_go/val_clip.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)

# class TouchTest(TouchBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/touch_and_go/test_clip.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)

class TouchTrain(TouchBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/touch_and_go/train_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", **kwargs)


class TouchValidation(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/val_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)

class TouchTest(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_identified_1k.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
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

class TouchTestBind(TouchBase):
    def __init__(self, flip_p=0., cond_modality=None, **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_identified_400.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
        self.text_root = '/home/fredyang/fredyang/stable-diffusion/data/touch_and_go/caption_final.csv'
        self.caption = pd.read_csv(self.text_root)
        self.cond_modality = cond_modality
    
    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)
        assert self.main_modality == 'touch'
        # load visual image
        example["image"] = self.load_test_img(example["touch_path_"])
        
        example["aux"] = self.load_test_img(example["image_path_"])
        
        if self.cond_modality == 'vision':
            example["aux_new"] = data.load_and_transform_vision_data([example["image_path_"]], device='cuda')
        elif self.cond_modality == 'text':
            # example["aux_new"] = self.load_test_img(example["image_path_"])
            text_path = self.caption.iloc[i, 2]
            
            example["aux_new"] = data.load_and_transform_text([text_path], device='cuda')
        else:
            print('no modality')
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
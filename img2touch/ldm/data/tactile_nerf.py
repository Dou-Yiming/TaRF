import os
import os.path as osp
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import json
import torchvision
import scipy
from ipdb import set_trace as st


class TactileNerfBase(Dataset):
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
                 set_type='train',
                 transform_flag=False,
                 rgb_flag=False,
                 depth_flag=False,
                 scale_list=['40_30']
                 ):
        assert main_modality in ['image', 'touch']
        self.data_paths = json_file
        self.data_root = data_root
        self.set_type = set_type

        # load candidates
        self.cand = []
        with open(json_file) as f:
            # [[obj, contact]]
            self.cand = json.load(f)[self.set_type]
            print(len(self.cand))
            self.cand = [[x[0].replace('_40_30',''), x[1]] for x in self.cand]
            # exclude_list = [
            #     'eecs_room_colmap',
            #     'eecs_3414_colmap',
            #     'eecs_office_2_colmap'
            # ]
            # self.cand = [x for x in self.cand if x[0] not in exclude_list]
            print(len(self.cand))
            # st()
            if self.set_type == 'train':
                self.cand *= 5
        self._length = len(self.cand)
        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.main_modality = main_modality
        
        self.scale_list = scale_list

        self.rotation = rotation
        self.flip = T.RandomHorizontalFlip(p=flip_p)
        self.transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=(0.9, 1.1),
                                         contrast=(0.9, 1.1),
                                         saturation=0.1)], p=0.8),
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=.1),
            T.RandomApply(
                [T.GaussianBlur(5, sigma=(.5, 1))], p=.5),
        ])
        self.transform_flag = transform_flag
        self.rgb_flag = rgb_flag
        self.depth_flag = depth_flag
        print("transform_flag: {}".format(transform_flag))

    def __len__(self):
        return self._length

    def load_rgb(self, scene, idx, scale):
        rgb_path = osp.join(self.data_root, 'vision',
                            '_'.join([scene, scale]), idx+'.jpg')
        rgb = Image.open(rgb_path)
        if not rgb.mode == "RGB":
            rgb = rgb.convert("RGB")
        # center crop
        crop = min(rgb.size[0], rgb.size[1])
        rgb = TF.center_crop(rgb, crop)
        # resize
        if self.size is not None:
            rgb = TF.resize(rgb, self.size, self.interpolation)
        rgb = TF.rotate(rgb, angle=90)
        if self.transform_flag == True:
            rgb = self.transform(rgb)
        # rotate
        rgb = np.array(rgb).astype(np.uint8)
        rgb = (rgb / 127.5 - 1.0).astype(np.float32)
        return rgb
    
    def load_depth(self, scene, idx, scale):
        depth_path = osp.join(self.data_root, 'depth',
                              '_'.join([scene, scale]), idx+'.npy')
        depth = np.load(depth_path)
        # center crop
        crop = min(depth.shape[0], depth.shape[1])
        depth = depth[(depth.shape[0] - crop) // 2:(depth.shape[0] + crop) // 2,
                      (depth.shape[1] - crop) // 2:(depth.shape[1] + crop) // 2, :]
        # resize
        depth = np.rot90(depth)
        depth = cv2.resize(depth, (self.size, self.size))
        depth = depth.reshape(self.size, self.size, 1).astype(np.float32)
        return depth
        
    def load_touch(self, scene, idx, scale):
        touch_path = osp.join(self.data_root, 'touch',
                                '_'.join([scene, scale]), idx+'.jpg')
        touch = Image.open(touch_path)
        if not touch.mode == "RGB":
            touch = touch.convert("RGB")
        # center crop
        crop = min(touch.size[0], touch.size[1])
        touch = TF.center_crop(touch, crop)
        # resize
        if self.size is not None:
            touch = TF.resize(touch, self.size, self.interpolation)
        # rotate
        touch = TF.rotate(touch, angle=90)
        touch = np.array(touch).astype(np.uint8)
        touch = (touch / 127.5 - 1.0).astype(np.float32)
        return touch
        
    def load_touch_bg(self, scene):
        touch_path = osp.join(self.data_root, 'touch_bg', '_'.join([scene, '40_30']), 'bg.jpg')
        touch = Image.open(touch_path)
        if not touch.mode == "RGB":
            touch = touch.convert("RGB")
        # center crop
        crop = min(touch.size[0], touch.size[1])
        touch = TF.center_crop(touch, crop)
        # resize
        if self.size is not None:
            touch = TF.resize(touch, self.size, self.interpolation)
        # rotate
        touch = TF.rotate(touch, angle=90)
        touch = np.array(touch).astype(np.uint8)
        touch = (touch / 127.5 - 1.0).astype(np.float32)
        return touch

    def __getitem__(self, i):
        scene, idx = self.cand[i]
        example = {}
        example['aux'] = []
        for scale in self.scale_list:
            if self.rgb_flag:
                example['aux'].append(self.load_rgb(scene, idx, scale))
            if self.depth_flag:
                example['aux'].append(self.load_depth(scene, idx, scale))
        example['aux'].append(self.load_touch_bg(scene))
        example['aux'] = np.concatenate(example['aux'], axis=-1)

        example['image'] = self.load_touch(scene, idx, '40_30')

        return example


class TouchTrain(TactileNerfBase):
    def __init__(self, flip_p=0., **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        super().__init__(json_file="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/", flip_p=flip_p, set_type='train',  **kwargs)


class TouchValidation(TactileNerfBase):
    def __init__(self, flip_p=0., **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        super().__init__(json_file="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/", flip_p=flip_p, set_type='val',  **kwargs)


class TouchTest(TactileNerfBase):
    def __init__(self, flip_p=0., only_aux=False, **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/split_interval.json",
        #                  data_root="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib", flip_p=flip_p, set_type='test', **kwargs)
        super().__init__(json_file="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/home/ymdou/TaRF/img2touch/data/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/", flip_p=flip_p, set_type='test', **kwargs)
        self.only_aux = only_aux

    def __getitem__(self, i):
        scene, idx = self.cand[i]
        
        example = {}
        example['aux'] = []
        for scale in self.scale_list:
            if self.rgb_flag:
                rgb = self.load_rgb(scene, idx, scale)
                rgb = rgb[None].transpose(0, 3, 1, 2)
                rgb = torch.from_numpy(rgb)
                example['aux'].append(rgb)
            if self.depth_flag:
                depth = self.load_depth(scene, idx, scale)
                depth = depth.reshape(1, 1, self.size, self.size).astype(np.float32)
                depth = torch.from_numpy(depth)
                example['aux'].append(depth)
        touch_bg = self.load_touch_bg(scene)
        touch_bg = touch_bg[None].transpose(0, 3, 1, 2)
        touch_bg = torch.from_numpy(touch_bg)
        example['aux'].append(touch_bg)
        example['aux'] = torch.cat(example['aux'], 1)
        
        touch = self.load_touch(scene, idx, '40_30')
        touch = touch[None].transpose(0, 3, 1, 2)
        touch = torch.from_numpy(touch)
        example['image'] = touch
        if self.only_aux:
            example["image"] = torch.zeros_like(example["image"])

        return example

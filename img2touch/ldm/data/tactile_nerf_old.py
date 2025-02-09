import os
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import json
import torchvision
import scipy


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
            if self.set_type == 'train':
                self.cand *= 5
        self._length = len(self.cand)
        self.data = {
            "image_path_": [os.path.join(self.data_root, 'vision', self.cand[i][0], self.cand[i][1] + '.jpg') for i in range(len(self.cand))],
            "touch_path_": [os.path.join(self.data_root,  'touch', self.cand[i][0], self.cand[i][1]+'.jpg') for i in range(len(self.cand))],
            "depth_path_": [os.path.join(self.data_root,  'depth', self.cand[i][0], self.cand[i][1]+'.npy') for i in range(len(self.cand))],
        }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality
        assert self.main_modality == 'touch'

        # rotation
        self.rotation = rotation

        self.transform = torchvision.transforms.Compose([
                                                        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=(0.9, 1.1),
                                                                                                                               contrast=(.9, 1.1),
                                                                                                                               saturation=0.1)], p=.8),
                                                        torchvision.transforms.RandomApply([torchvision.transforms.Grayscale(num_output_channels=3)], p=.1),
                                                        torchvision.transforms.RandomApply(
                                                            [torchvision.transforms.GaussianBlur(5, sigma=(.5, 1))], p=.5),
                                                        ])
        self.transform_flag = transform_flag
        self.rgb_flag = rgb_flag
        self.depth_flag = depth_flag
        print("transform_flag: {}".format(transform_flag))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)

        # load visual image
        if self.main_modality == 'image':
            image = Image.open(example["image_path_"])
        elif self.main_modality == 'touch':
            image = Image.open(example["touch_path_"])
            bg_path = example["touch_path_"].replace('touch/','touch_bg/')
            bg_path = bg_path.replace(bg_path.split('/')[-1], 'bg.jpg')
            bg = Image.open(bg_path)
            if not bg.mode == "RGB":
                bg = bg.convert("RGB")
        else:
            print('No matched modality')
            exit()
        if not image.mode == "RGB":
            image = image.convert("RGB")
        

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        bg = np.array(bg).astype(np.uint8)
        
        crop = min(img.shape[0], img.shape[1])
        crop_bg = min(bg.shape[0], bg.shape[1])
        
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        bg = bg[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        
        image = Image.fromarray(img)
        bg = Image.fromarray(bg)
        
        image = transforms.functional.rotate(image, angle=90)
        bg = transforms.functional.rotate(bg, angle=90)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
            bg = bg.resize((self.size, self.size),
                                 resample=self.interpolation)

        # image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        bg = np.array(bg).astype(np.uint8)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["bg"] = (bg / 127.5 - 1.0).astype(np.float32)

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
        # if self.transform_flag == True:
        #     crop_ratio = np.random.uniform(0.6, 1.0)
        #     crop = int(crop_ratio*crop)
            
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = transforms.functional.rotate(image, angle=90)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        image = self.flip(image)
        if self.transform_flag == True:
            image = self.transform(image)

        image = np.array(image).astype(np.uint8)
        example["aux"] = (image / 127.5 - 1.0).astype(np.float32)
        
        # load depth
        if self.depth_flag:
            depth_map = np.load(example["depth_path_"])
            depth_map = np.clip(depth_map, 0, 5)
            crop = min(depth_map.shape[0], depth_map.shape[1])
            h, w, = depth_map.shape[0], depth_map.shape[1]
            depth_map = depth_map[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2,:]
            depth_map = np.rot90(depth_map)
            depth_map = cv2.resize(depth_map, (self.size, self.size))
            depth_map = depth_map.reshape(self.size, self.size, 1).astype(np.float32)
            example['aux'] = np.concatenate((example['aux'], depth_map), axis=-1)
        example['aux'] = np.concatenate((example['aux'], example['bg']), axis=-1)
        if not self.rgb_flag:
            example['aux'] = example['aux'][:,:,3:]
            
        # if self.transform_flag == True:
        #     # rotation = random.uniform(0, 360)
        #     # example['aux'] = scipy.ndimage.rotate(example['aux'], rotation, axes=(0,1), reshape=False)
        #     # example['image'] = scipy.ndimage.rotate(example['image'], rotation, axes=(0,1), reshape=False)
        #     # example["aux"] = np.rot90(example["aux"], rotation, axes=(0,1))
        #     # example["image"] = np.rot90(example["image"], rotation, axes=(0,1))
        #     hroi_flip, vert_flip = random.choice([(False, False), (True, False), (False, True), (True, True)])
        #     if hroi_flip:
        #         example["aux"] = np.fliplr(example["aux"])
        #         example["image"] = np.fliplr(example["image"])
        #     if vert_flip:
        #         example["aux"] = np.flipud(example["aux"])
        #         example["image"] = np.flipud(example["image"])
        #     example["aux"] = np.ascontiguousarray(example["aux"])
        #     example["image"] = np.ascontiguousarray(example["image"])
    
            
        return example

    def load_test_img(self, path):
        image = Image.open(path).convert("RGB")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image = transforms.functional.rotate(image, angle=90)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        # image = self.flip(image)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.
    
    def load_test_depth(self, path):
        depth_map = np.load(path)
        depth_map = np.clip(depth_map, 0, 5)
        crop = min(depth_map.shape[0], depth_map.shape[1])
        h, w, = depth_map.shape[0], depth_map.shape[1]
        depth_map = depth_map[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2,:]
        depth_map = np.rot90(depth_map)
        depth_map = cv2.resize(depth_map, (self.size, self.size))
        depth_map = depth_map.reshape(1, 1, self.size, self.size).astype(np.float32)
        depth_map = torch.from_numpy(depth_map)
        return depth_map

class TouchTrain(TactileNerfBase):
    def __init__(self, flip_p=0., **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final/", flip_p=flip_p, set_type='train',  **kwargs)


class TouchValidation(TactileNerfBase):
    def __init__(self, flip_p=0., **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final/", flip_p=flip_p, set_type='val',  **kwargs)


class TouchTest(TactileNerfBase):
    def __init__(self, flip_p=0., only_aux=False, **kwargs):
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/data/tactile_nerf_v2/split.json",
        # super().__init__(json_file="/nfs/turbo/coe-ahowens/datasets/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/split_interval.json",
        #                  data_root="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib", flip_p=flip_p, set_type='test', **kwargs)
        super().__init__(json_file="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/tactile_nerf/vision_touch_pairs_tactile_nerf_final/split_interval.json",
                         data_root="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/tactile_nerf/vision_touch_pairs_tactile_nerf_final", flip_p=flip_p, set_type='test', **kwargs)
        self.only_aux = only_aux

    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)

        # load visual image
        if self.main_modality == 'image':
            if self.only_aux:
                example["image"] = self.load_test_img(example["touch_path_"])
                example["image"] = torch.zeros_like(example["image"])
            else:
                example["image"] = self.load_test_img(example["image_path_"])
        elif self.main_modality == 'touch':
            if self.only_aux:
                example["image"] = self.load_test_img(example["image_path_"])
                example["image"] = torch.zeros_like(example["image"])
            else:
                example["image"] = self.load_test_img(example["touch_path_"])
        else:
            print('No matched modality')
            exit()

        # load aux image
        if self.main_modality == 'image':
            example["aux"] = self.load_test_img(example["touch_path_"])
        elif self.main_modality == 'touch':
            example["aux"] = self.load_test_img(example["image_path_"])
            bg_path = example["touch_path_"].replace('touch/','touch_bg/')
            bg_path = bg_path.replace(bg_path.split('/')[-1], 'bg.jpg')
            bg = self.load_test_img(bg_path)
        else:
            print('No matched modality')
            exit()
        
        if self.depth_flag:
            depth_map = self.load_test_depth(example["depth_path_"])
            # print(depth_map.shape, example['aux'].shape)
            example['aux'] = torch.cat((example['aux'], depth_map), 1)
        example['aux'] = torch.cat((example['aux'], bg), 1)
        if not self.rgb_flag:
            example['aux'] = example['aux'][:,3:]
            
        # rotation = random.choice([0, 90, 180, 270])
        # example["aux"][:,:4] = transforms.functional.rotate(example["aux"][:,:4], rotation)
        
        # rotate aux
        if self.rotation != 0:
            example["aux"] = transforms.functional.rotate(
                example["aux"], angle=self.rotation)

        return example

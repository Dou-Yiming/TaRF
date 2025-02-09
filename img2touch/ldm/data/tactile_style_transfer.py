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
from ipdb import set_trace as st

GELSLIM_MEAN = torch.tensor([-0.0082, -0.0059, -0.0066])
GELSLIM_STD = torch.tensor([0.0989, 0.0746, 0.0731])
BUBBLES_MEAN = torch.tensor([0.00382])
BUBBLES_MIN, BUBBLES_MAX = -0.0116, 0.0225
BUBBLES_STD = torch.tensor([0.00424])


class TactileStyleTransferBase(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 main_modality='bubble',
                 seg_mask=False,
                 mask_size=None,
                 rotation=0,
                 set_type='train',
                 transform_flag=False,
                 rgb_flag=False,
                 depth_flag=False,
                 ):
        assert main_modality in ['bubble', 'gelslim']
        self.data_paths = json_file
        self.data_root = data_root
        self.set_type = set_type
        train_tools = ['pattern_01_2_lines_angle_1',
                       'pattern_03_2_lines_angle_3',
                       'pattern_04_3_lines_angle_1',
                       'pattern_05_3_lines_angle_2',
                       'pattern_06_5_lines_angle_1',
                       'pattern_07_curves_degree_30_radios_10',
                       'pattern_09_curves_degree_120_radios_10',
                       'pattern_10_curves_degree_150_radios_10',
                       'pattern_11_curves_degree_30_radios_20',
                       'pattern_12_curves_degree_45_radios_20',
                       'pattern_14_curves_degree_150_radios_20',
                       'pattern_15_circle',
                       'pattern_17_ellipse_2',
                       'pattern_18_hex_1',
                       'pattern_20_hex_3',
                       'pattern_31_rod']

        test_tools = ['pattern_02_2_lines_angle_2',
                      'pattern_05_3_lines_angle_2',
                      'pattern_08_curves_degree_45_radios_10',
                      'pattern_13_curves_degree_120_radios_20',
                      'pattern_16_ellipse_1',
                      'pattern_19_hex_2',
                      'test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen'
                      ]
        # load candidates
        self.cand = []
        with open(json_file) as f:
            # [[obj, contact]]
            self.cand = json.load(f)[self.set_type]
            # self.cand = json.load(open(json_file))['train'] + json.load(open(json_file))['val']
            if self.set_type == 'train':
                self.cand = [x for x in self.cand if x[0] in train_tools]
                self.cand *= 5
            elif self.set_type == 'test':
                self.cand = [x for x in self.cand if x[0] in test_tools]
            # self.cand = json.load(f)['train']
            # if self.set_type == 'train':
        self._length = len(self.cand)
        if self.set_type == 'test':
            self.data = {
                "bubble_path_": [os.path.join(self.data_root, 'bubbles', 'bubbles_testing_data_processed_flipped_2',
                                              'bubble_style_transfer_dataset_bubbles_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
                "gelslim_path_": [os.path.join(self.data_root,  'gelslims', 'gelslim_testing_data_processed_flipped_2',
                                               'gelslim_style_transfer_dataset_gelslim_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
            }
            # self.data = {
            #     "bubble_path_": [os.path.join(self.data_root, 'bubbles', 'bubbles_training_data_processed_flipped',
            #                                   'bubble_style_transfer_dataset_bubbles_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
            #     "gelslim_path_": [os.path.join(self.data_root,  'gelslims', 'gelslim_training_data_processed_flipped',
            #                                    'gelslim_style_transfer_dataset_gelslim_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
            # }
        else:
            self.data = {
                "bubble_path_": [os.path.join(self.data_root, 'bubbles', 'bubbles_training_data_processed_flipped',
                                              'bubble_style_transfer_dataset_bubbles_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
                "gelslim_path_": [os.path.join(self.data_root,  'gelslims', 'gelslim_training_data_processed_flipped',
                                               'gelslim_style_transfer_dataset_gelslim_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
            }
        # self.data = {
        #         "bubble_path_": [os.path.join(self.data_root, 'bubbles', 'bubbles_training_data_processed_flipped',
        #                                       'bubble_style_transfer_dataset_bubbles_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
        #         "gelslim_path_": [os.path.join(self.data_root,  'gelslims', 'gelslim_training_data_processed_flipped',
        #                                        'gelslim_style_transfer_dataset_gelslim_'+self.cand[i][0], self.cand[i][1]+'.pt') for i in range(len(self.cand))],
        #     }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality

        # rotation
        self.rotation = rotation

        self.gelslim_transform = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    (self.size, self.size), interpolation=self.interpolation),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1)], p=0.8),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.GaussianBlur(5, sigma=(0.5, 1))], p=0.5),
                torchvision.transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD),
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    (self.size, self.size), interpolation=self.interpolation),
                torchvision.transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD),
            ]),
            'test': torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    (self.size, self.size), interpolation=self.interpolation),
                torchvision.transforms.Normalize(GELSLIM_MEAN, GELSLIM_STD),
            ])}
        self.bubble_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (self.size, self.size), interpolation=self.interpolation),
            # torchvision.transforms.Normalize(BUBBLES_MEAN, BUBBLES_STD),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)

        finger_index = random.randint(0, 1)

        bubble = torch.load(example["bubble_path_"])
        bubble = bubble['bubble_imprint'][finger_index]
        bubble = 2 * (bubble - BUBBLES_MIN) / (BUBBLES_MAX - BUBBLES_MIN) - 1
        bubble = torchvision.transforms.functional.rotate(bubble, 180)
        bubble = self.bubble_transform(bubble).repeat(3, 1, 1)
        bubble = bubble.permute(1, 2, 0)
        bubble = np.array(bubble).astype(np.float32)
        assert bubble.max() <= 1 and bubble.min() >= - \
            1, print(bubble.max(), bubble.min())

        gelslim = torch.load(example["gelslim_path_"])
        gelslim = gelslim['gelslim'][finger_index] - \
            gelslim['gelslim_ref'][finger_index]
        gelslim = self.gelslim_transform[self.set_type](gelslim)
        gelslim = gelslim.permute(1, 2, 0)
        gelslim = np.array(gelslim).astype(np.float32)

        if self.main_modality == 'bubble':
            example['aux'] = gelslim
            example['image'] = bubble
        elif self.main_modality == 'gelslim':
            example['aux'] = bubble
            example['image'] = gelslim

        # if self.set_type == 'train':
        #     rotation = random.uniform(-180, 180)
        #     example['aux'] = scipy.ndimage.rotate(
        #         example['aux'], rotation, axes=(0, 1), reshape=False)
        #     example['image'] = scipy.ndimage.rotate(
        #         example['image'], rotation, axes=(0, 1), reshape=False)
        #     hroi_flip, vert_flip = random.choice(
        #         [(False, False), (True, False), (False, True), (True, True)])
        #     if hroi_flip:
        #         example["aux"] = np.fliplr(example["aux"])
        #         example["image"] = np.fliplr(example["image"])
        #     if vert_flip:
        #         example["aux"] = np.flipud(example["aux"])
        #         example["image"] = np.flipud(example["image"])
        #     example["aux"] = np.ascontiguousarray(example["aux"])
        #     example["image"] = np.ascontiguousarray(example["image"])

        return example


class TactileStyleTransferTrain(TactileStyleTransferBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/home/ymdou/stable-diffusion/data/tactile_style_transfer/split.json",
                         data_root="/home/ymdou/stable-diffusion/data/tactile_style_transfer", flip_p=flip_p, set_type='train',  **kwargs)


class TactileStyleTransferValidation(TactileStyleTransferBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/home/ymdou/stable-diffusion/data/tactile_style_transfer/split.json",
                         data_root="/home/ymdou/stable-diffusion/data/tactile_style_transfer", flip_p=flip_p, set_type='val',  **kwargs)


class TactileStyleTransferTest(TactileStyleTransferBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(json_file="/home/ymdou/stable-diffusion/data/tactile_style_transfer/split_new_test.json",
                         data_root="/home/ymdou/stable-diffusion/data/tactile_style_transfer", flip_p=flip_p, set_type='test',  **kwargs)

    def __getitem__(self, i):
        example = dict((k, self.data[k][i]) for k in self.data)

        # finger_index = random.randint(0, 1)
        finger_index = 0

        bubble = torch.load(example["bubble_path_"])
        index = '_'.join(example["bubble_path_"].split(
            '/')[-2:]).replace('.pt', '').replace('bubble_style_transfer_dataset_bubbles_', '')
        example['index'] = index
        bubble = bubble['bubble_imprint'][finger_index]
        # bubble = bubble['bubble_imprint']
        bubble = 2 * (bubble - BUBBLES_MIN) / (BUBBLES_MAX - BUBBLES_MIN) - 1
        bubble = torchvision.transforms.functional.rotate(bubble, 180)
        bubble = self.bubble_transform(bubble).repeat(3, 1, 1)
        bubble = bubble.unsqueeze(0)

        gelslim = torch.load(example["gelslim_path_"])
        gelslim = gelslim['gelslim'][finger_index] - \
            gelslim['gelslim_ref'][finger_index]
        # gelslim = gelslim['gelslim'] - gelslim['gelslim_ref']
        gelslim = self.gelslim_transform[self.set_type](gelslim)
        gelslim = gelslim.unsqueeze(0)

        if self.main_modality == 'bubble':
            example['aux'] = gelslim
            example['image'] = bubble
        elif self.main_modality == 'gelslim':
            example['aux'] = bubble
            example['image'] = gelslim

        return example

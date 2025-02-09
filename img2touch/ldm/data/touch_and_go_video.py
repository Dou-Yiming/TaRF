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
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 main_modality='image',
                 reference_length=3,
                 skip_frame=0,
                 seg_mask=False,
                 mask_size=None,
                 reflectance=False,
                 cond_mode='hybrid'
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
            "class": [int(self.image_label[i]) for i in range(len(self.image_paths))],
            "root_path_": [os.path.join(self.data_root, self.sub_folder[i]) for i in range(len(self.image_paths))]
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.main_modality = main_modality

        self.reference_length = reference_length
        assert reference_length % 2 == 1, "reference_length must be a odd number"
        self.skip_frame = skip_frame

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
        
        # reflectance
        self.reflectance = reflectance
        self.reflectance_root = '/nfs/turbo/coe-ahowens/datasets/touch_and_go/intrinsic'
        self.cond_mode = cond_mode

    def __len__(self):
        return self._length

    def process_img(self, image):
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
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image


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
        relative_file_path_ = example['relative_file_path_']
        current_idx = int(os.path.basename(relative_file_path_)[:-4])
        offset = (self.reference_length - 1 ) / 2
        aux_frame_list = []

        start_idx = int(current_idx - offset * (self.skip_frame + 1))

        # print(example["image_path_"], relative_file_path_, current_idx)
        for i in range(self.reference_length):
            idx = start_idx + i + self.skip_frame
            idx = str(idx).rjust(10,'0') + '.jpg'
            
            if self.main_modality == 'image':
                load_path = os.path.join(example["root_path_"], 'gelsight_frame', idx)
            elif self.main_modality == 'touch':
                load_path = os.path.join(example["root_path_"], 'video_frame', idx)
            else:
                print('No matched modality')
                exit()
            
            temp_img = Image.open(load_path)
            
            if not temp_img.mode == "RGB":
                temp_img = temp_img.convert("RGB")

            temp_img = self.process_img(temp_img)
            aux_frame_list.append(temp_img)
        
        # print(aux_frame_list[0].shape)
        aux_frame_list = np.concatenate(aux_frame_list, axis=-1)
        example["aux"] = aux_frame_list

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
            seg_mask = torch.permute(seg_mask, (2, 0, 1))
            seg_mask = self.flip(seg_mask)
            
            if self.size is not None:
                # seg_mask = torch.permute(seg_mask, (2, 0, 1))
                seg_mask = self.torch_resize(seg_mask)
            # seg_mask_dup = seg_mask
            
            if self.mask_size is not None:
                # seg_mask = self.mask_resize(seg_mask)
                seg_mask = seg_mask.float()
                seg_mask = self.max_pool(seg_mask)
                seg_mask = seg_mask.int()
                seg_mask = seg_mask.bool()
             
            example['seg_mask'] = seg_mask[0,:,:]
        
        # class_label
        example['class_label'] = np.array(example['class'] + 1)
        
        # reflectance
        if self.reflectance == True:
            reflectance_path = os.path.join(self.reflectance_root, example['relative_file_path_'][:15] + '_' + example['relative_file_path_'][16:-4] + '-r' + '.jpg')
            
            image = Image.open(reflectance_path)

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
            reflectance = (image / 127.5 - 1.0).astype(np.float32)


            if self.reflectance == True:
                example['reflectance'] = reflectance
                example['aux'] = np.concatenate([reflectance, example['aux']], axis=2)
                # print(example['aux'].shape)
            else:
                print('wrong reflectance mode')
                exit()
            

        
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
        super().__init__(txt_file="data/touch_and_go/train_clip_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", **kwargs)


class TouchValidation(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/val_clip_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)

class TouchTest(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_clip_identified.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
    
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
        relative_file_path_ = example['relative_file_path_']
        current_idx = int(os.path.basename(relative_file_path_)[:-4])
        offset = (self.reference_length - 1 ) / 2
        aux_frame_list = []

        start_idx = int(current_idx - offset * (self.skip_frame + 1))

        # print(example["image_path_"], relative_file_path_, current_idx)
        for i in range(self.reference_length):
            idx = start_idx + i + self.skip_frame
            idx = str(idx).rjust(10,'0') + '.jpg'
            
            if self.main_modality == 'image':
                load_path = os.path.join(example["root_path_"], 'gelsight_frame', idx)
            elif self.main_modality == 'touch':
                load_path = os.path.join(example["root_path_"], 'video_frame', idx)
            else:
                print('No matched modality')
                exit()
            
            temp_img = self.load_test_img(load_path)
            aux_frame_list.append(temp_img)
        
        example["aux"] = torch.cat(aux_frame_list, axis=1)

        if self.reflectance == True:
            reflectance_path = os.path.join(self.reflectance_root, example['relative_file_path_'][:15] + '_' + example['relative_file_path_'][16:-4] + '-r' + '.jpg')
            reflectance = self.load_test_img(reflectance_path)

            if self.reflectance == True:
                # print(reflectance.shape, example['aux'].shape)
                example['aux'] = torch.cat([reflectance, example['aux']], dim=1)
                example['reflectance'] = reflectance
                # print(example['aux'].shape)
                # print(example['aux'].shape)
            else:
                print('wrong reflectance mode')
                exit()
        
        # class_label
        example['class_label'] = torch.from_numpy(np.array(example['class'] + 1))
            


        return example


class TDIS(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_clip.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
    
    def __getitem__(self, i):
        # set up A, B
        A_index = i
        B_index = random.randint(0,self._length)
        example = dict((k, self.data[k][A_index]) for k in self.data)
        example2 = dict((k, self.data[k][B_index]) for k in self.data)
        
        # load visual image
        example["imageA"] = self.load_test_img(example["image_path_"])
        example["gelsightA"] = self.load_test_img(example["touch_path_"])
        example["imageB"] = self.load_test_img(example2["image_path_"])
        example["gelsightB"] = self.load_test_img(example2["touch_path_"])


        return example


class TDIS_ref_video(TouchBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/touch_and_go/test_clip.txt", data_root="/nfs/turbo/coe-ahowens/datasets/touch_and_go/final_dataset", flip_p=flip_p, **kwargs)
    
    def __getitem__(self, i):
        # set up A, B
        A_index = i
        B_index = random.randint(0,self._length)
        example = dict((k, self.data[k][A_index]) for k in self.data)
        example2 = dict((k, self.data[k][B_index]) for k in self.data)
        
        # load visual image
        example["imageA"] = self.load_test_img(example["image_path_"])
        example["gelsightA"] = self.load_test_img(example["touch_path_"])
        example["imageB"] = self.load_test_img(example2["image_path_"])
        # example["gelsightB"] = self.load_test_img(example2["touch_path_"])


         # load aux image
        relative_file_path_ = example2['relative_file_path_']
        current_idx = int(os.path.basename(relative_file_path_)[:-4])
        offset = (self.reference_length - 1 ) / 2
        aux_frame_list = []

        start_idx = int(current_idx - offset * (self.skip_frame + 1))

        # print(example["image_path_"], relative_file_path_, current_idx)
        for i in range(self.reference_length):
            idx = start_idx + i + self.skip_frame
            idx = str(idx).rjust(10,'0') + '.jpg'
            
            if self.main_modality == 'image':
                load_path = os.path.join(example2["root_path_"], 'gelsight_frame', idx)
            elif self.main_modality == 'touch':
                load_path = os.path.join(example2["root_path_"], 'video_frame', idx)
            else:
                print('No matched modality')
                exit()
            
            temp_img = self.load_test_img(load_path)
            aux_frame_list.append(temp_img)
        
        example["gelsightB"] = torch.cat(aux_frame_list, axis=1)

        # load ref
        if self.reflectance == True:
            reflectance_path = os.path.join(self.reflectance_root, example['relative_file_path_'][:15] + '_' + example['relative_file_path_'][16:-4] + '-r' + '.jpg')
            reflectance = self.load_test_img(reflectance_path)

            if self.reflectance == True:
                # print(reflectance.shape, example['aux'].shape)
                if self.cond_mode == 'hybrid':
                    example['aux'] = torch.cat([reflectance, example['gelsightB']], dim=1)
                # print(example['aux'].shape)
                # print(example['aux'].shape)
                example['reflectance'] = reflectance
            else:
                print('wrong reflectance mode')
                exit()

        


        return example
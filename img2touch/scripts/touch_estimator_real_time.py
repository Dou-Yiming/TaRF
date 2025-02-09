import argparse
import os
import os.path as osp
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from torchvision import transforms
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.models.resnet import resnet18, resnet50
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ipdb import set_trace as st
import shutil

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def map_back(image):
    image = 127.5 * (image + 1.0) / 255.0
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to load rgbd from",
        default=""
    )
    parser.add_argument(
        "--bg_path",
        type=str,
        nargs="?",
        help="path to load background from",
        default=""
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=""
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--diffusion_ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ranking_rgb_enc_ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ranking_tac_enc_ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    return opt

class RankingEncoder(nn.Module):
    def __init__(self, feature_dim=32,
                 model_type='resnet18',
                 **kwargs):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(512, feature_dim)
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Linear(2048, feature_dim)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        feat_norm = torch.norm(features, dim=1)
        features /= feat_norm.view(features.shape[0], 1)
        return features

    def encode(self, im: np.ndarray):
        '''
        takes in an image and returns the resnet18 features
        '''
        im = im.unsqueeze(0)
        with torch.no_grad():
            features = self(im)
        return features

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(
            save_name, map_location=torch.device('cpu')))

class TouchEstimator:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load_model()
        self.load_ranking_model()

    def load_model(self):
        config = OmegaConf.load(f"{self.opt.config}")
        model = load_model_from_config(config, f"{self.opt.diffusion_ckpt}")
        model = model.to(self.device)
        model.eval()
        if self.opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        self.model = model
        self.sampler = sampler
        
    def load_ranking_model(self):
        print('Loading re-ranking model...')
        self.ranking_preprocess = {
            'rgb': T.Compose([
                T.ToTensor(),
                T.Resize(128),
            ]),
            'tac': T.Compose([
                T.ToTensor(),
                T.Resize(128),
            ])}

        rgb_encoder = RankingEncoder(feature_dim=32, model_type='resnet50')
        tac_encoder = RankingEncoder(feature_dim=32, model_type='resnet50')

        rgb_ckpt_path = self.opt.ranking_rgb_enc_ckpt
        tac_ckpt_path = self.opt.ranking_tac_enc_ckpt

        rgb_encoder.load(rgb_ckpt_path)
        tac_encoder.load(tac_ckpt_path)
        
        rgb_encoder.eval()
        tac_encoder.eval()

        rgb_encoder = rgb_encoder.cuda()
        tac_encoder = tac_encoder.cuda()
        
        self.ranking_rgb_encoder = rgb_encoder
        self.ranking_tac_encoder = tac_encoder
    
    def load_rgbdb(self, rgb_path_list, depth_path_list, bg_path):
        rgbdb = []
        for rgb_path, depth_path in zip(rgb_path_list, depth_path_list):
            # load rgb
            rgb = Image.open(rgb_path).convert("RGB")
            rgb = np.array(rgb).astype(np.uint8)
            crop = min(rgb.shape[0], rgb.shape[1])
            h, w, = rgb.shape[0], rgb.shape[1]
            rgb = rgb[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]
            rgb = Image.fromarray(rgb)
            rgb = rgb.resize((256, 256), resample=PIL.Image.BICUBIC)
            rgb = np.array(rgb).astype(np.float32) / 255.0
            rgb = rgb[None].transpose(0, 3, 1, 2)
            rgb = torch.from_numpy(rgb)
            rgb = 2. * rgb - 1.
            
            # load depth
            depth_map = np.load(depth_path)
            crop = min(depth_map.shape[0], depth_map.shape[1])
            h, w, = depth_map.shape[0], depth_map.shape[1]
            depth_map = depth_map[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2,:]
            depth_map = cv2.resize(depth_map, (256, 256))
            depth_map = np.clip(depth_map, 0.0, 5.0)
            depth_map = depth_map / 5.0 * 2.0 - 1.0
            depth_map = depth_map.reshape(1, 1, 256, 256).astype(np.float32)
            depth_map = torch.from_numpy(depth_map)
            
            rgbdb.append(rgb)
            rgbdb.append(depth_map)
        
        # load bg
        bg = Image.open(bg_path).convert("RGB")
        bg = np.array(bg).astype(np.uint8)
        crop = min(bg.shape[0], bg.shape[1])
        h, w, = bg.shape[0], bg.shape[1]
        bg = bg[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        bg = Image.fromarray(bg)
        bg = transforms.functional.rotate(bg, angle=90)
        bg = bg.resize((256, 256), resample=PIL.Image.BICUBIC)
        bg = np.array(bg).astype(np.float32) / 255.0
        bg = bg[None].transpose(0, 3, 1, 2)
        bg = torch.from_numpy(bg)
        bg = 2. * bg - 1.
        rgbdb.append(bg)
        
        
        rgbdb = torch.cat(rgbdb, 1)
        
        return rgbdb

    def estimate(self, rgb_path_list, depth_path_list, bg_path):
        outdir = self.opt.outdir
        os.makedirs(outdir, exist_ok=True)
        start_code = None
        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    uc = None
                    rgbdb = self.load_rgbdb(rgb_path_list, depth_path_list, bg_path).to(self.device)
                    prompts = torch.cat([rgbdb for _ in range(self.opt.n_samples)])
                    c = self.model.get_learned_conditioning(prompts)
                    shape = (self.model.channels, self.model.image_size, self.model.image_size)
                    samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=self.opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=self.opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=self.opt.ddim_eta,
                                                    x_T=start_code)
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image = x_samples_ddim

                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    
                    origin_images = [map_back(rgbdb[:,:3])]
                    origin_name = ['rgb']

                    for index, x_sample in enumerate(origin_images):
                        x_sample = 255. * rearrange(x_sample.squeeze().cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        
                        img.save(os.path.join(outdir, "{}.png".format(origin_name[index])))
                    
                    generated_image_count = 0
                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(outdir, f"{generated_image_count:02}.png"))
                        generated_image_count += 1
                        
                    rgb = Image.open(os.path.join(outdir, "{}.png".format(origin_name[index])))
                    rgb = TF.rotate(rgb, angle=-90)
                    rgb = self.ranking_preprocess['rgb'](rgb).cuda()
                    rgb = rgb.unsqueeze(0)
                    
                    sample_tacs = [Image.open(os.path.join(outdir, f"{gen_i:02}.png")) for gen_i in range(generated_image_count)]
                    sample_tacs = [TF.rotate(tac, angle=-90) for tac in sample_tacs]
                    sample_tacs = torch.stack([self.ranking_preprocess['tac'](tac) for tac in sample_tacs]).cuda()
                    print("re-ranking...")
                    with torch.no_grad():
                        rgb_feats = self.ranking_rgb_encoder(rgb)
                        tac_feats = self.ranking_tac_encoder(sample_tacs)
                        dot_prods = rgb_feats.mm(tac_feats.t()).cpu()
                    print('max dot product: {:.4f}'.format(torch.max(dot_prods)))
                    print('min dot product: {:.4f}'.format(torch.min(dot_prods)))
                    max_sample_id = int(torch.argmax(dot_prods).cpu())
                    shutil.copy(os.path.join(outdir, f"{max_sample_id:02}.png"), 
                                os.path.join(outdir, f"best.png"))
                        

def main():
    opt = parse_args()
    seed_everything(opt.seed)
    touch_estimator = TouchEstimator(opt)
    
    indir = opt.indir
    rgb_dir = os.path.join(indir, 'rgb')
    depth_dir = os.path.join(indir, 'depth')
    rgb_path_list = [
        osp.join(rgb_dir, '40_50.png'),
        osp.join(rgb_dir, '0_40.png')
    ]
    depth_path_list = [
        osp.join(depth_dir, '40_50.npy'),
        osp.join(depth_dir, '0_40.npy')
    ]
    bg_path = opt.bg_path
    touch_estimator.estimate(rgb_path_list, depth_path_list, bg_path)
    
    latest_rgb_mtime = os.path.getmtime(rgb_path_list[-1])
    latest_depth_mtime = os.path.getmtime(depth_path_list[-1])
    print('Start listening for new rgb and depth images...')
    while True:
        cur_rgb_mtime = os.path.getmtime(rgb_path_list[-1])
        cur_depth_mtime = os.path.getmtime(depth_path_list[-1])
        if cur_rgb_mtime == latest_rgb_mtime or cur_depth_mtime == latest_depth_mtime:
            time.sleep(0.1)
            continue
        latest_rgb_mtime = cur_rgb_mtime
        latest_depth_mtime = cur_depth_mtime
        print('New rgb and depth images detected, estimating touch...')
        touch_estimator.estimate(rgb_path_list, depth_path_list, bg_path)
        print('Touch estimation finished.')


if __name__ == "__main__":
    main()

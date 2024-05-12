import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
import pytorch_lightning as pl

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

from ldm.modules.encoders.resnet_cmc import MyResNetsCMC
from torchvision import models


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))
    
# 创建cmc encoder
class CMCEncoder(pl.LightningModule):
    def __init__(self,
                 model_name='resnet50v1',
                 model_path=None,
                 layer=5,
                 modality='touch',
                 device='cuda',
                 reference_length=1,
                 not_load=False):
        super().__init__()
        assert modality in ['touch','image']
        self.model = MyResNetsCMC(model_name, modality, reference_length)
        self.layer = layer

        if not_load == True:
            print('Trained CMC from scratch')
        else:  
            if model_path != None:
                print('==> loading pre-trained CMC encoder from {}'.format(model_path))
                sd = torch.load(model_path, map_location="cpu")
                self.load_state_dict(sd, strict=False)
                print('==> done')
            else:
                print('No pretrained CMC encoder')
                exit()

    def forward(self,x):
        feat = self.model(x, layer=self.layer)
        if self.layer == 5:
            feat = torch.flatten(feat, start_dim=2, end_dim=3)
            feat = torch.permute(feat, (0, 2, 1))
            # print(feat.shape)
            # exit()
        elif self.layer == 6:
            feat = torch.unsqueeze(feat, 1)
        return feat

    def encode(self, x):
        return self(x)

class CMCEncoder_ref(pl.LightningModule):
    def __init__(self,
                 model_name='resnet50v1',
                 model_path=None,
                 layer=5,
                 modality='touch',
                 device='cuda',
                 reference_length=1,
                 not_load=False):
        super().__init__()
        assert modality in ['touch','image']
        self.model = MyResNetsCMC(model_name, modality, reference_length)
        self.layer = layer

        if not_load == True:
            print('Trained CMC from scratch')
        else:  
            if model_path != None:
                print('==> loading pre-trained CMC encoder from {}'.format(model_path))
                sd = torch.load(model_path, map_location="cpu")
                self.load_state_dict(sd, strict=False)
                print('==> done')
            else:
                print('No pretrained CMC encoder')
                exit()

    def forward(self,x):
        feat = self.model(x, layer=self.layer)
        if self.layer == 5:
            feat = torch.flatten(feat, start_dim=2, end_dim=3)
            feat = torch.permute(feat, (0, 2, 1))
            print(feat.shape)
            exit()
        elif self.layer == 6:
            feat = torch.unsqueeze(feat, 1)
        return feat

    def encode(self, x):
        return self(x)

class ResnetEncoder(pl.LightningModule):
    def __init__(self,
                 model_name='resnet18',):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 14x14 or 16x16
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1) # 28x28 or 32x32
        
    def forward(self,x):
        out1 = self.features(x)
        out2 = self.deconv1(F.relu(out1))
        out3 = self.deconv2(F.relu(out2))
        return out3

    def encode(self, x):
        return self(x)
    
class ResnetRGBDEncoder(pl.LightningModule):
    def __init__(self,
                 model_name='resnet18',):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            # rgbdb
            # self.resnet.conv1 = nn.Conv2d(4+3, 64, 7, 2, 3)
            self.resnet.conv1 = nn.Conv2d(4+4+3, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(4+4+4+3, 64, 7, 2, 3)
            # rgbb
            # self.resnet.conv1 = nn.Conv2d(3+3, 64, 7, 2, 3)
            # db
            # self.resnet.conv1 = nn.Conv2d(1+3, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 14x14 or 16x16
            self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1) # 28x28 or 32x32
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)
            self.resnet.conv1 = nn.Conv2d(4+4+3, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(1+3, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(3+3, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(),
                self.deconv1,
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                self.deconv2,
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                self.deconv3,
            )
            
        
    def forward(self,x):
        out = self.features(x)
        out = self.deconv(out)
        # out1 = self.upsample(out)
        # out2 = self.deconv1(F.leaky_relu(out1))
        # out3 = self.deconv2(F.leaky_relu(out2))
        # out4 = self.deconv3(F.leaky_relu(out3))
        
        return out

    def encode(self, x):
        return self(x)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
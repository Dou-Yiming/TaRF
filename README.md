# Tactile-Augmented Radiance Fields (CVPR 2024)

### [Dataset](https://www.dropbox.com/scl/fo/xusq5gvwauwakred27q9o/AHfcGs56Dv1ZCeitEM3-8_A?rlkey=07efzqehs918no80yi4jrokvh&st=0g4s2ioy&dl=0) · [Checkpoints](https://www.dropbox.com/scl/fo/h3jn0io2cnjz9m7n4i7l1/AIe4VzbKOusFFe7-ic_zHX0?rlkey=1q6337nku09i0mw1q2sjjk5os&st=vsab9jal&dl=0) · [Website](https://dou-yiming.github.io/TaRF/) · [Paper](https://arxiv.org/abs/2405.04534)

##### [Yiming Dou](https://dou-yiming.github.io/), [Fengyu Yang](https://fredfyyang.github.io/), [Yi Liu](), [Antonio Loquercio](https://antonilo.github.io/), [Andrew Owens](https://andrewowens.com/)

![teaser](./assets/figs/tarf.gif)

## Installation
#### 1. Install Nerfstudio modules
##### 1.1. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
##### 1.2. Clone this repo
`git clone https://github.com/Dou-Yiming/TaRF.git`

##### 1.3. Install tarf modules as a python package

```sh
cd TaRF/nerfstudio_modules
python -m pip install -e .
```
##### 1.4. Run `ns-install-cli`

##### 1.5. Check the install
Run `ns-train -h`: you should see a list of "subcommands" with tarf included among them.

#### 2. Install Diffusion modules
##### 2.1 Create another Conda environment
`conda create --name ldm -y python=3.8`

##### 2.2 Install ldm
```sh
cd TaRF/img2touch
python -m pip install -e .
```

## Using TaRF

## Train your own TaRF

Coming soon!

#### Bibtex

If you find TaRF useful, please conside citing:

```
@inproceedings{dou2024tactile,
  title={Tactile-augmented radiance fields},
  author={Dou, Yiming and Yang, Fengyu and Liu, Yi and Loquercio, Antonio and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26529--26539},
  year={2024}
}
```
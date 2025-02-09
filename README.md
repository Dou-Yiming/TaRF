# Tactile-Augmented Radiance Fields (CVPR 2024)

### [Dataset](https://www.dropbox.com/scl/fo/xusq5gvwauwakred27q9o/AHfcGs56Dv1ZCeitEM3-8_A?rlkey=07efzqehs918no80yi4jrokvh&st=0g4s2ioy&dl=0) · [Checkpoints](https://www.dropbox.com/scl/fo/h3jn0io2cnjz9m7n4i7l1/AIe4VzbKOusFFe7-ic_zHX0?rlkey=1q6337nku09i0mw1q2sjjk5os&st=vsab9jal&dl=0) · [Website](https://dou-yiming.github.io/TaRF/) · [Paper](https://arxiv.org/abs/2405.04534)

##### [Yiming Dou](https://dou-yiming.github.io/), [Fengyu Yang](https://fredfyyang.github.io/), [Yi Liu](), [Antonio Loquercio](https://antonilo.github.io/), [Andrew Owens](https://andrewowens.com/)

![teaser](./assets/figs/tarf.gif)

## Installation
### 1. Install Nerfstudio modules
##### 1.1. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/quickstart/installation.html) to install dependencies and create an environment.

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

### 2. Install Diffusion modules
##### 2.1 Create another Conda environment
`conda create --name ldm -y python=3.8`

##### 2.2 Install ldm
```sh
cd TaRF/img2touch
python -m pip install -e .
```

## Using TaRF

### 1. Prepare data and pretrained models

##### 1.1 Download COLMAP database

Download the COLMAP databases (including images and camera poses of 13 scenes) from [this](https://www.dropbox.com/scl/fo/chyl14skmqqqlqn6qjn32/AEWhshTnqKn7IAE_QIxCqZM?rlkey=kcic3d0p5wyb3zad5x4srt1vc&st=2zjjhoq7&dl=0) link, then extract them:

```sh
cd TaRF/nerfstudio_modules
mkdir data
cd data
tar -xvf {dir of the downloaded colmap folder}/*.tar.gz ./
```

##### 1.2 Download pertrained models

Download the pretrained NeRF models from [this](https://www.dropbox.com/scl/fo/kgexp5j82t2obbfzwtol5/AJZ_nVEgWI7oE_tZrmMXw2o?rlkey=bx7o6ow6csmjsvg3svlgwjo9i&st=e6yocux0&dl=0) link, then extract them:

```sh
cd TaRF/nerfstudio_modules
mkdir outputs
cd outputs
tar -xvf {dir of the downloaded NeRF folder}/*.tar.gz ./
```

Download the pretrained Diffusion models from [this](https://www.dropbox.com/scl/fi/5n9vx5991ev8av5l6ca2e/pretrained_models.tar.gz?rlkey=gdbkyot3at0hrr76np0hu220n&st=7krfblmx&dl=0) link, then extract them:

```sh
tar -xvf {dir of the downloaded Diffusion folder}/pretrained_models.tar.gz ./
mv pretrained_models TaRF/img2touch
mv TaRF/img2touch/pretrained_models/first_stage_model.ckpt TaRF/img2touch/models/first_stage_models/kl-f8/model.ckpt
```

### 2. Run & interact with TaRF

##### 2.1 Launch Nerfstudio viewer

First, pick a scene from {conference_room, clothes, office_1, stairs, table_outdoor, workroom, bench, snow, bus_stop, office_2, chair_outdoor, table_indoor lounge}

Next, launch the Nerfstudio viewer:

```sh
cd TaRF/nerfstudio_modules
ns-viewer --load-config ./outputs/{scene}_colmap/tarf/test/config.yml --vis viewer --viewer.max-num-display-images 64
```

This will provide you with a link to the Nerfstudio viewer.

##### 2.2 Interact with the TaRF

Now you may interact with the TaRF in the browser with the following steps:

1. Click the "Touch on Scene" button at the bottom-right corner.
2. Click the point you want to touch in the TaRF.

This will give you the egocentric RGBD signal of the clicked point, saved in `TaRF/nerfstudio_modules/outputs/touch_estimation_input_cache`.

### 3. Estimate touch signals

First, change the `bg_path` argument in `TaRF/img2touch/scripts/bash_scripts/run_touch_estimator_real_time.sh` to `touch_bg/{scene}_colmap_40_50/bg.jpg`.

Next, launch real-time touch estimator on **another GPU**:

```sh
cd TaRF/img2touch
bash scripts/bash_scripts/run_touch_estimator_real_time.sh
```

The tactile signals can now be estimated in real-time whenever a new point is clicked, the results will be saved at `TaRF/img2touch/outputs/touch_estimator_real_time/best.png`.

## Train your own TaRF

##### 1. Train touch estimator

```sh
cd TaRF/img2touch
bash scripts/bash_scripts/train_touch_estimator.sh
```

Coming soon!

### Bibtex

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
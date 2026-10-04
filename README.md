<div align="center">

<h1>
  Visual2Echo Compositional Contrastive Learning (V2E-CCL): Binaural Knowledge distilled network for Depth prediction
</h1>

### CVPR 2026 Findings 🇺🇸

<a href="https://ailab.space">Nazrul Ismail</a><sup>1</sup>,
Owais Ahmed Malik<sup>2</sup>,
<a href="https://ailab.space/">Ong Wee Hong</a><sup>1</sup>

<sup>1</sup>Robotics and Intelligent Systems Laboratory (RoboLab), School of Digital Science, Universiti Brunei Darussalam

<sup>2</sup>Atlantic Technological University

[![CVF](https://img.shields.io/badge/CVF-Paper-005A9C?logo=ieee&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Ismail_Visual2Echo_Compositional_Contrastive_Learning_V2E-CCL_Binaural_Knowledge_Distilled_Network_for_CVPRF_2026_paper.pdf)
</div>

---

<p align="center">
  <img src="assets/EchoNet - pipeline_final.png" alt="V2E-CCL Overview" width="85%">
  <br>
  <em>Overview of the Visual2Echo Compositional Contrastive Learning (V2E-CCL) framework for cross-modal binaural depth prediction.</em>
</p>

## Abstract
> Depth estimation from audio is an active area of research with applications in robotics and assistive technologies, yet remains underexplored compared to vision-based approaches. Echo reflections inherently capture physically-grounded information, including object displacement, shape, and material properties. Inspired by biological echolocators like bats, we tackle the challenging problem of estimating depth and material properties using only binaural echoes from single audio chirps. Recent work has addressed audio depth estimation by augmenting other modalities or applying cross-modal knowledge distillation from vision to audio. We propose Visual2Echo Compositional Contrastive Learning (V2E-CCL), a knowledge distillation framework that bridges the visual-auditory domain gap through two key components: a Compositional Embedding (CE) module that refines vision teacher latent features by incorporating audio cues, and a Compositional Contrastive Learning (CCL) module that aligns cross-modal spatial representations in a unified latent space. Extensive evaluation shows our method achieves RMSE improvements of 28\% on the Replica dataset and 48\% on the Matterport3D dataset compared to prior audio-only approaches, while demonstrating consistent gains (14\%) across different teacher architectures including modern foundation models.
> 

---

## Environment Setup

**Tested on:** Python 3.12, PyTorch 2.2.1, NVIDIA RTX 4090

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively with conda:
```bash
conda create -n v2e python=3.10
conda activate v2e
pip install -r requirements.txt
```

---

## Datasets
**Replica-VisualEchoes** can be obtained from [here](https://github.com/facebookresearch/VisualEchoes). Below are the commands to download.

```
# rgb-depth pairs of 4 different resolutions
# dictionary is in the format of {scene:{(location, orientation): {'rgb':rgb_image, 'depth':depth_map}}}
wget http://dl.fbaipublicfiles.com/VisualEchoes/rgb_depth/scene_observations_128.pkl

# echo responses for the 3ms sweep signal at all navigable locations
#    ├── echoes_navigable                          
#    │       └── [scene]                         (scene name)
#    │           └── [sweep_sound]               (name of the source signal)
#    │               └── [angle]                 (agent's orientation)
#    │                   └── location_index.wav  (agent's location)
wget http://dl.fbaipublicfiles.com/VisualEchoes/echoes_navigable.tar.gz
```

**MatterportEchoes (MP3D)** is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of MP3D dataset. 

### MatterportEchoes (mp3d) — recommended
- **Images/Depth:** per-scene `.pkl` files organised as `{img_path}/{split}.pkl` (and a `scenes/` sub-directory of per-scene pickles)
- **Audio:** `.wav` files at `{audio_path}/{scene}/{audio_type}/{orientation}/{location}.wav`
- **Sampling rate:** 16 kHz
- **Metadata splits:** `dataset/metadata/mp3d/mp3d_scenes_{train,val,test}.txt`

Expected directory layout:
```
mp3d/
  mp3d_split_wise/       # --img_path
    train.pkl
    val.pkl
    test.pkl
    scenes/
      {scene_id}.pkl
      ...
  echoes_navigable/      # --audio_path
    {scene_id}/
      {audio_type}/
        {orientation}/
          {location}.wav
```

### Replica
- **Images/Depth:** single monolithic `.pkl` file (`scene → location → orientation → {rgb, depth}`)
- **Audio:** same layout as mp3d
- **Sampling rate:** 44.1 kHz
- **Metadata splits:** `dataset/metadata/replica/replica_{train,val,test}.txt`

---

## Pretrained Checkpoints

Place pretrained weights in `checkpoints_pretrained/`:

| File | Description |
|------|-------------|
| `material_pre_trained_minc.pth` | Material classifier (23 MINC classes), **required** |
| `rgbdepth_mp3d.pth` / `rgbdepth_replica.pth` | Frozen RGB U-Net teacher |

---

## Training

`--dataset` fixes the STFT and depth range: mp3d (16 kHz, win/hop 32, `max_depth` 10), replica (44.1 kHz, win 256, hop 32, `max_depth` 14.104), biosonar (320 kHz, win 128, hop 64, 20–100 kHz band).

### MP3D / Replica

```bash
python3 train_ccl.py \
  --dataset mp3d \
  --img_path /path/to/mp3d/mp3d_split_wise \
  --metadatapath dataset/metadata/mp3d \
  --audio_path /path/to/mp3d/echoes_navigable \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --exp_name my_experiment --batchSize 64 --validation_on \
  --niter 201 --epoch_save_freq 250 --display_freq 400 --validation_freq 800 \
  --freeze_nets --depth_loss_type berhu --use_ipd \
  --lr_audio 2.5e-4 --weight_decay 1e-4 \
  --ccl_temperature 0.1 --lambda_mat 0.0 --lambda_ccl_depth 0.05
```

For Replica use `--dataset replica --img_path /path/to/scene_observations_128.pkl --metadatapath dataset/metadata/replica`.

### EchoScene (biosonar) with a cached MoGe-2 teacher

Cache the teacher once per split (rows follow dataset order):

```bash
python3 precompute_teacher_latents.py \
  --dataset biosonar --img_path $DATA --audio_path $DATA --biosonar_rgb_dir $DATA \
  --rgb_teacher moge_v2 --moge_model_id Ruicheng/moge-2-vits-normal \
  --moge_resolution_level 0 --moge_num_tokens 256 --moge_use_fp16 \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --teacher_cache_path latents/moge_train.h5 --mode_ext train
# repeat with --mode_ext val --val_include_test --teacher_cache_path latents/moge_valtest.h5
```

Then train (ldc2 recipe):

```bash
python3 train_ccl.py \
  --dataset biosonar --img_path $DATA --audio_path $DATA --biosonar_rgb_dir $DATA \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --backbone Resnet18 --decoder_spatial_entry --audio_norm_type groupnorm --audio_norm_groups 8 \
  --batvision_img_size 128 --depth_resize_method nearest --max_depth 8 \
  --audio_crop_pre_roll 0.002 --audio_length 0.075 --log_spectrogram --audio_normalize --use_specaugment \
  --batchSize 32 --niter 30 --validation_on --validation_freq 385 --val_include_test \
  --freeze_nets --depth_loss_type silog --lambda_depth 1.0 --lambda_grad 0.25 --lambda_ssim 0.5 \
  --lambda_feat_std 1.0 --feat_std_gamma 0.5 --audio_decoder_dropout 0.2 \
  --lr_audio 5e-4 --weight_decay 0 --cosine_T_max 11520 --deterministic --seed 1 \
  --rgb_teacher moge_v2 --moge_model_id Ruicheng/moge-2-vits-normal --moge_cache_enc_dim_out 384 \
  --teacher_cache_path latents/moge_train.h5 --validation_cache_path latents/moge_valtest.h5 \
  --cache_freeze_teacher_proj --lambda_teacher_depth 0.5 --teacher_depth_align \
  --lambda_ccl_depth 0.5 --lambda_ldc 0.2 --lambda_cc 0.1 \
  --lambda_mat 0 --lambda_ccl_mat 0 --lambda_ct 0 --ct_am_scale 0 \
  --exp_name echoscene_ldc2
```

Checkpoints go to `checkpoint/{exp_name}/{dataset}/`, TensorBoard logs to `runs/{exp_name}/`.

---

## Evaluation

`train_ccl.py --validation_on` reports `ABS_REL`, `RMSE`, `LOG10`, `MAE`, `DELTA1/2/3` (δ < 1.25, 1.25², 1.25³), computed in `util/util.py:compute_errors()`.

---

## Architecture

- **Audio stream** (`SimpleAudioDepthNet`): BinauralResNet18/34 or MALFNet on STFT magnitudes (+ sin/cos IPD with `--use_ipd`, ILD with `--use_ild`) → U-Net decoder → `[B, 1, 128, 128]`
- **Visual teacher**: frozen RGB U-Net (`RGBDepthNet`) or MoGe-2 (`--rgb_teacher moge_v2`, live or cached)
- **Material stream** (`MaterialPropertyNet`): frozen ResNet18 → 23 MINC classes
- **CCL**: `CompositionalEmbedding` heads align audio features with teacher depth and material features; `ProjectionHead`s for the contrastive term

---

## Project Structure

```
train_ccl.py                  training
precompute_teacher_latents.py teacher feature cache (HDF5)
eval_pretrained.py            evaluate a Legacy audio checkpoint on Replica
split_pkl_by_scene.py         split a monolithic pkl into per-scene pkls
models/                       networks, MoGe teacher, losses, ModelBuilder
data_loader/                  Replica/MP3D, EchoScene (biosonar), cached-latent datasets
options/                      CLI options
dataset/metadata/             scene splits
```

## Acknowledgement
Some codes in this repo are adapted from [VisualEchoes](https://github.com/facebookresearch/VisualEchoes.git) and [Beyond Image to Depth](https://github.com/krantiparida/beyond-image-to-depth).  We thank the authors for making their code and ideas publicly available.

## Citation
If you find this work or the code useful in your research, please consider citing our paper:
```bibtex
@InProceedings{Ismail_2026_CVPR,
    author    = {Ismail, Nazrul and Malik, Owais Ahmed and Hong, Ong Wee},
    title     = {Visual2Echo Compositional Contrastive Learning (V2E-CCL): Binaural Knowledge Distilled Network for Depth Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6019-6028}
}

<div align="center">

<h1>
  Visual2Echo Compositional Contrastive Learning (V2E-CCL): Binaural Knowledge distilled network for Depth prediction
</h1>

### CVPR 2026 Findings

<a href="https://ailab.space">Nazrul Ismail</a><sup>1</sup>,
Owais Ahmed Malik<sup>2</sup>,
<a href="https://ailab.space/">Ong Wee Hong</a><sup>1</sup>

<sup>1</sup>Robotics and Intelligent Systems Laboratory (RoboLab), School of Digital Science, Universiti Brunei Darussalam

<sup>2</sup>Atlantic Technological University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)]()
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Checkpoint-yellow?logo=huggingface&logoColor=yellow)](R)

</div>


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

**MatterportEchoes (MP3D) ** is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of MP3D dataset. 

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

Place all pretrained weights in `checkpoints_pretrained/`:

| File | Description |
|------|-------------|
| `material_pre_trained_minc.pth` | ResNet18 material classifier (23 MINC classes) — **required** |
| `rgbdepth_mp3d.pth` | Frozen RGB teacher for mp3d |
| `rgbdepth_replica.pth` | Frozen RGB teacher for Replica |
| `audiodepth_mp3d.pth` | Pretrained audio depth model for mp3d |
| `audiodepth_replica.pth` | Pretrained audio depth model for Replica |

---

## Training

### MP3D (CCL variant)

```bash
python3 train_ccl.py \
  --validation_on \
  --dataset mp3d \
  --batchSize 64 \
  --img_path /path/to/mp3d/mp3d_split_wise \
  --metadatapath dataset/metadata/mp3d \
  --audio_path /path/to/mp3d/echoes_navigable \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --exp_name my_experiment \
  --niter 201 --epoch_save_freq 250 --display_freq 400 --validation_freq 800 \
  --freeze_nets --max_depth 10 \
  --depth_loss_type berhu --use_ipd \
  --lr_audio 2.5e-4 --weight_decay 1e-4 \
  --ccl_temperature 0.1 --lambda_mat 0.0 --lambda_ccl_depth 0.05
```

### Replica (CCL variant)

```bash
python3 train_ccl.py \
  --validation_on \
  --dataset replica \
  --batchSize 64 \
  --img_path /path/to/replica/scene_observations_128.pkl \
  --metadatapath dataset/metadata/replica \
  --audio_path /path/to/echoes/echoes_navigable \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --exp_name my_experiment \
  --niter 201 --epoch_save_freq 250 --display_freq 400 --validation_freq 800 \
  --freeze_nets --max_depth 5 \
  --depth_loss_type berhu --use_ipd \
  --lr_audio 2.5e-4 --weight_decay 1e-4 \
  --ccl_temperature 0.1 --lambda_mat 0.0 --lambda_ccl_depth 0.05
```

Checkpoints are saved to `checkpoint/{exp_name}/`. TensorBoard logs go to `runs/{exp_name}/`.

---

## Evaluation / Testing

```bash
python3 test.py \
  --dataset mp3d \
  --batchSize 128 \https://ai.stanford.edu/~rhgao/https://ai.stanford.edu/~rhgao/
  --img_path /path/to/mp3d/mp3d_split_wise \
  --metadatapath dataset/metadata/mp3d \
  --audio_path /path/to/mp3d/echoes_navigable \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --audio_std_weight_dir checkpoint/my_experiment \
  --audio_std_weight_pth audiodepth_mp3d_epoch_200.pth \
  --exp_name test_run \
  --max_depth 10
```

To evaluate a shipped pretrained checkpoint directly:

```bash
python3 eval_pretrained.py \
  --img_path /path/to/replica/scene_observations_128.pkl \
  --audio_path /path/to/echoes/echoes_navigable \
  --metadatapath dataset/metadata/replica \
  --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
  --weights checkpoints_pretrained/audiodepth_replica.pth \
  --max_depth 5
```

### Reported metrics

`ABS_REL`, `RMSE`, `LOG10`, `MAE`, `DELTA1` (δ < 1.25), `DELTA2` (δ < 1.25²), `DELTA3` (δ < 1.25³)

---

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mp3d` | `mp3d` or `replica` |
| `--max_depth` | `10` | Max depth clamp (10 m mp3d, 5 m replica) |
| `--depth_loss_type` | `log` | `log`, `berhu`, `silog`, `l1`, `l2` |
| `--use_ipd` | off | Add interaural phase difference as 3rd channel |
| `--log_spectrogram` | off | Log-compress spectrogram magnitudes |
| `--lr_audio` | `1e-4` | Audio backbone + decoder learning rate |
| `--ccl_temperature` | `1.0` | CCL contrastive temperature |
| `--lambda_ccl_depth` | `0.3` | Weight for CCL audio-depth loss |
| `--lambda_mat` | `0.7` | Weight for material classification loss |
| `--lambda_teacher_depth` | `0.5` | Weight for RGB teacher distillation |
| `--freeze_nets` | off | Freeze RGB teacher and material nets |
| `--backbone` | `Resnet18` | `Resnet18` or `Resnet34` |
| `--audio_length` | `0.06` | Audio window length in seconds |
| `--audio_nfft` | `512` | STFT n_fft (affects frequency resolution) |

---

## Architecture

Three parallel streams merged at the loss level:

- **Audio stream** (`SimpleAudioDepthNet`): BinauralResNet18 backbone on 2-channel (or 3-channel with `--use_ipd`) STFT spectrograms → UNet decoder → `[B, 1, 128, 128]` depth map
- **Visual stream** (`RGBDepthNet`): frozen RGB UNet teacher
- **Material stream** (`MaterialPropertyNet`): frozen pretrained ResNet18 → 23 MINC material classes

Loss: `L = L_depth + λ_mat·L_mat + λ_ccl·L_CCL + λ_ct·L_CT + λ_teacher·L_teacher`

Input spectrograms:
- **mp3d:** `[B, 2, 257, 121]` at 16 kHz, n_fft=512, hop=128
- **Replica:** `[B, 2, 257, 83]` at 44.1 kHz, n_fft=512, hop=32, win=256

---

## Project Structure

```
Visual2Echo/
  train_ccl.py              # main training script (CCL variant)
  test.py                   # evaluation script
  eval_pretrained.py        # quick eval of shipped checkpoints
  models/
    networks.py             # SimpleAudioDepthNet, UNet decoder, CCL heads
    backbone.py             # BinauralResNet18/34
    audioVisual_model_ccl.py# forward pass orchestration
    models.py               # ModelBuilder factory
  data_loader/
    audio_visual_dataset.py # data loading, STFT spectrogram generation
  options/
    base_options.py         # shared CLI arguments
    train_options.py        # training-specific CLI arguments
  dataset/metadata/         # train/val/test scene lists
  checkpoints_pretrained/   # pretrained weights
  util/util.py              # compute_errors() metric computation
  results.tsv               # experiment log (not tracked by git)
```

## Acknowledgement
Some codes in this repo are adapted from [VisualEchoes](https://github.com/facebookresearch/VisualEchoes.git) and [Beyond Image to Depth](https://github.com/krantiparida/beyond-image-to-depth).  We thank the authors for making their code and ideas publicly available.

## Citation
If you find this work or the code useful in your research, please consider citing our paper:
```bibtex
To  appear

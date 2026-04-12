"""Compositional Contrastive Learning (CCL) training script.

Trains the audio depth estimation network using:
- Depth supervision (LogDepth / BerHu / SILog)
- Teacher depth distillation from RGB teacher
- Material classification distillation
- Cross-modal contrastive learning (cosine CCL)

Usage:
    python3 train_ccl.py \
        --dataset mp3d \
        --img_path /path/to/mp3d_split_wise \
        --audio_path /path/to/echoes_navigable \
        --metadatapath dataset/metadata/mp3d \
        --init_material_weight checkpoints_pretrained/material_pre_trained_minc.pth \
        --init_audiodepth_weight checkpoint/ssl_pretrain/mp3d/audiodepth_ssl_pretrained.pth \
        --batchSize 64 --niter 100 \
        --lambda_depth 1.0 --lambda_mat 0.7 --lambda_ct 0.05 \
        --validation_on --exp_name ccl_train
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model_ccl import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import TextWrite, compute_errors, AverageMeter

import numpy as np
from models import criterion


def freeze_weights(net):
    for param in net.parameters():
        param.requires_grad = False


def evaluate(trainer, dataset_val, opt, writer, epoch):
    losses = []
    errors = []
    depth_losses = AverageMeter()
    mat_losses = AverageMeter()
    losses_ct_ad = AverageMeter()
    losses_ct_am = AverageMeter()
    losses_ccl_ad = AverageMeter()
    losses_ccl_am = AverageMeter()

    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            output, losses_val = trainer.forward(val_data)
            _, depth_loss, mat_loss, ccl_audiodepth_loss, ccl_audiomat_loss, \
                loss_ct_ad, loss_ct_am, _ = losses_val

            depth_losses.update(depth_loss, opt.batchSize)
            mat_losses.update(mat_loss, opt.batchSize)

            total_loss = (trainer.lambda_depth * depth_loss) + (trainer.lambda_mat * mat_loss)
            losses.append(total_loss.item())

            losses_ct_ad.update(loss_ct_ad.item(), opt.batchSize)
            losses_ct_am.update(loss_ct_am.item(), opt.batchSize)
            losses_ccl_ad.update(ccl_audiodepth_loss.item(), opt.batchSize)
            losses_ccl_am.update(ccl_audiomat_loss.item(), opt.batchSize)

            depth_gt_val = output["depth_gt"].float()
            audio_depth_val = output["audio_depth"].float()
            for idx in range(audio_depth_val.shape[0]):
                mask = ((depth_gt_val[idx] > 0) &
                        (depth_gt_val[idx] < opt.max_depth)).cpu().numpy()
                errors.append(compute_errors(
                    depth_gt_val[idx].cpu().numpy(),
                    audio_depth_val[idx].cpu().numpy(), mask=mask))

    mean_loss = sum(losses) / len(losses)
    mean_errors = np.array(errors).mean(0)

    print('Loss: {:.3f}, RMSE: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}'.format(
        mean_loss, mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

    val_errors = {
        'ABS_REL/STD': mean_errors[0],
        'RMSE/STD': mean_errors[1],
        'DELTA1/STD': mean_errors[2],
        'DELTA2/STD': mean_errors[3],
        'DELTA3/STD': mean_errors[4],
    }

    if writer:
        writer.add_images('val/audio_pred_depth', output["audio_depth"], epoch, dataformats="NCHW")
        if output["img_depth"] is not None:
            writer.add_images('val/rgb_pred_depth', output["img_depth"], epoch, dataformats="NCHW")
        writer.add_images('val/depth_gt', output["depth_gt"], epoch, dataformats="NCHW")
        writer.add_scalar('val/mat_loss', mat_losses.avg, epoch)
        writer.add_scalar('val/depth_loss', depth_losses.avg, epoch)
        writer.add_scalar('val/ccl_audiomat_loss', losses_ccl_am.avg, epoch)
        writer.add_scalar('val/ccl_audiodepth_loss', losses_ccl_ad.avg, epoch)

    return mean_loss, val_errors


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, nets, opt, augment=False):
        self.nets = nets
        self.model = AudioVisualModel(self.nets, opt)

        if len(opt.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        self.model.to(opt.device)

        self.opt = opt
        self.augment = augment

        # Loss weights
        self.lambda_depth = getattr(opt, 'lambda_depth', 1.0)
        self.lambda_mat = getattr(opt, 'lambda_mat', 0.)
        self.lambda_ccl_depth = getattr(opt, 'lambda_ccl_depth', 0.)
        self.lambda_ccl_mat = 0.
        self.lambda_ccl = max(self.lambda_ccl_depth, self.lambda_ccl_mat)
        self.lambda_ct = getattr(opt, 'lambda_ct', 0.)
        self.lambda_tv = 0.0
        self.lambda_multiscale = getattr(opt, 'lambda_multiscale', 0.0)
        self.lambda_laplacian = getattr(opt, 'lambda_laplacian', 0.0)
        self.lambda_ssim = getattr(opt, 'lambda_ssim', 0.0)
        self.lambda_grad = getattr(opt, 'lambda_grad', 0.0)
        self.lambda_teacher_depth = getattr(opt, 'lambda_teacher_depth', 0.25)

        if self.lambda_laplacian > 0:
            self.laplacian_loss = criterion.LaplacianDepthLoss(max_depth=opt.max_depth)
        else:
            self.laplacian_loss = None
        if self.lambda_ssim > 0:
            self.ssim_loss = criterion.SSIMLoss()
        else:
            self.ssim_loss = None

        # Loss functions
        _max_depth = opt.max_depth
        _depth_loss_type = getattr(opt, 'depth_loss_type', 'log')
        if _depth_loss_type == 'silog':
            self.loss_criterion = criterion.SILogLoss(scaling_factor=1.0, max_depth=_max_depth)
        elif _depth_loss_type == 'l1':
            self.loss_criterion = criterion.L1Loss()
        elif _depth_loss_type == 'l2':
            self.loss_criterion = criterion.L2Loss()
        elif _depth_loss_type == 'berhu':
            self.loss_criterion = criterion.Berhuloss(
                threshold=getattr(opt, 'berhu_threshold', 0.2), max_depth=_max_depth)
        else:
            self.loss_criterion = criterion.LogDepthLoss(max_depth=_max_depth)

        self.loss_mat_criterion = criterion.DistillationLoss()
        self.ccl_audiomat_criterion = criterion.DistillationLoss()

        _ccl_depth_loss = getattr(opt, 'ccl_depth_loss', 'log')
        if _ccl_depth_loss == 'berhu':
            self.ccl_audiodepth_criterion = criterion.Berhuloss(
                threshold=0.2, max_depth=_max_depth)
        else:
            self.ccl_audiodepth_criterion = criterion.LogDepthLoss(
                zero_depth_weight=0.9, max_depth=_max_depth)

        _ccl_temp = getattr(opt, 'ccl_temperature', 1.0)
        self.criterion_ccl_cosine = criterion.CosineCCLLoss(temperature=_ccl_temp)
        self.criterion_ct_am = criterion.NCELoss(temperature=1., head='MatNet')
        self.criterion_ct_ad = criterion.CosineCCLLoss(temperature=_ccl_temp)

        self.optimizer = self._create_optimizer()
        self._use_fp16 = getattr(opt, 'use_fp16', False)
        if self._use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self._accum_steps = max(1, getattr(opt, 'accumulation_steps', 1))
        self._accum_step = 0

        # LR scheduling
        _cosine_steps = getattr(opt, 'cosine_T_max', 0)
        _plateau_factor = getattr(opt, 'lr_plateau_factor', 0.0)
        if _plateau_factor > 0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=_plateau_factor,
                patience=getattr(opt, 'lr_plateau_patience', 5), min_lr=1e-7, verbose=True)
            self._plateau_scheduler = True
        elif _cosine_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=_cosine_steps, eta_min=1e-6)
            self._plateau_scheduler = False
        else:
            self.scheduler = None
            self._plateau_scheduler = False

        _warmup_steps = getattr(opt, 'warmup_steps', 0)
        if _warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_steps)
        else:
            self.warmup_scheduler = None
        self._warmup_step_count = 0
        self._warmup_steps = _warmup_steps

        # EMA
        _ema_decay = getattr(opt, 'ema_decay', 0.0)
        self.ema_decay = _ema_decay
        if _ema_decay > 0:
            import copy
            self.ema_net = copy.deepcopy(self._student_net().net_audio)
            for p in self.ema_net.parameters():
                p.requires_grad_(False)
        else:
            self.ema_net = None

        m = self._student_net()
        if getattr(opt, 'teacher_cache_path', ''):
            m.net_rgbdepth.cpu()
            m.net_material.cpu()
        torch.cuda.empty_cache()

    def _create_optimizer(self):
        m = self._student_net()
        backbone_params = list(m.net_audio.feature_extraction.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in backbone_ids]
        lr_bb = getattr(self.opt, 'lr_backbone_main', self.opt.lr_audio * 0.1)
        param_groups = [
            {'params': backbone_params, 'lr': lr_bb},
            {'params': other_params, 'lr': self.opt.lr_audio},
        ]
        if self.opt.optimizer == 'sgd':
            return torch.optim.SGD(param_groups, momentum=self.opt.beta1,
                                   weight_decay=self.opt.weight_decay)
        return torch.optim.Adam(param_groups, betas=(self.opt.beta1, 0.999),
                                weight_decay=self.opt.weight_decay)

    def decrease_learning_rate(self, decay_factor=0.94):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_factor

    def swap_ema(self):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            if self.ema_net is None:
                yield
                return
            net = self._student_net().net_audio
            import copy
            live_state = copy.deepcopy(net.state_dict())
            net.load_state_dict(self.ema_net.state_dict())
            try:
                yield
            finally:
                net.load_state_dict(live_state)
        return _ctx()

    def _student_net(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _to_device(self, data):
        return {k: v.to(self.opt.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

    def forward(self, data):
        data = self._to_device(data)
        _zero = torch.zeros(1, device=self.opt.device)

        compute_ccl = (self.lambda_ccl != 0.0)
        compute_ct = (self.lambda_ct != 0.0)

        output = self.model.forward(data, compute_ccl=compute_ccl, compute_ct=compute_ct)

        depth_gt = output['depth_gt']
        depth_predicted = output['audio_depth']
        audio_mat_class = output['audio_mat_class']
        material_class = output['material_class']
        audio_feat = output['audio_feat']
        material_feat = output['material_feat']

        depth_loss = self.loss_criterion(depth_predicted, depth_gt)
        mat_loss = (self.loss_mat_criterion([audio_mat_class, material_class], None)
                    if self.lambda_mat != 0.0 else _zero)

        # Teacher depth distillation
        img_depth = output['img_depth']
        if self.lambda_teacher_depth > 0.0 and img_depth is not None:
            valid_mask = (depth_gt > 0) & (depth_gt < self.opt.max_depth)
            if valid_mask.any():
                if getattr(self.opt, 'berhu_teacher', False):
                    teacher_depth_loss = self.loss_criterion(
                        depth_predicted * valid_mask.float(),
                        img_depth.detach() * valid_mask.float())
                else:
                    teacher_depth_loss = F.smooth_l1_loss(
                        depth_predicted[valid_mask], img_depth.detach()[valid_mask])
            else:
                teacher_depth_loss = _zero
        else:
            teacher_depth_loss = _zero

        # CCL losses
        if compute_ccl:
            ccl_audiodepth = output['ccl_audiodepth']
            ccl_audiomat = output['ccl_audiomat']
            ccl_audiodepth_loss = (self.ccl_audiodepth_criterion(ccl_audiodepth, depth_gt)
                                   if ccl_audiodepth is not None else _zero)
            ccl_audiomat_loss = self.ccl_audiomat_criterion(
                [ccl_audiomat, material_class], None)
        else:
            ccl_audiodepth_loss = ccl_audiomat_loss = _zero

        # Contrastive losses
        if compute_ct:
            aud_proj_feat = output['aud_proj_feat']
            img_proj_feat = output['img_proj_feat']
            loss_ct_ad = self.criterion_ct_ad(aud_proj_feat, img_proj_feat)
            af = F.adaptive_avg_pool2d(audio_feat.detach(), (1, 1)).flatten(1)
            mf = F.adaptive_avg_pool2d(material_feat, (1, 1)).flatten(1)
            loss_ct_am = self.criterion_ccl_cosine(af, mf)
        else:
            loss_ct_am = loss_ct_ad = _zero

        # Aggregate losses
        total_ccl_loss = (self.lambda_ccl_depth * ccl_audiodepth_loss +
                          self.lambda_ccl_mat * ccl_audiomat_loss)
        total_ct_loss = self.lambda_ct * (loss_ct_ad + loss_ct_am)

        # TV smoothness
        if self.lambda_tv > 0 and depth_predicted is not None:
            diff_x = torch.abs(depth_predicted[:, :, :, 1:] - depth_predicted[:, :, :, :-1])
            diff_y = torch.abs(depth_predicted[:, :, 1:, :] - depth_predicted[:, :, :-1, :])
            tv_loss = diff_x.mean() + diff_y.mean()
        else:
            tv_loss = _zero

        # Multi-scale depth loss
        if self.lambda_multiscale > 0 and depth_predicted is not None:
            ms_loss = _zero
            for scale in [2, 4]:
                ds_pred = F.avg_pool2d(depth_predicted, scale, stride=scale)
                ds_gt = F.avg_pool2d(depth_gt, scale, stride=scale)
                ms_loss = ms_loss + self.loss_criterion(ds_pred, ds_gt)
            multiscale_depth_loss = self.lambda_multiscale * ms_loss
        else:
            multiscale_depth_loss = _zero

        # Laplacian edge-aware loss
        if self.lambda_laplacian > 0 and self.laplacian_loss is not None:
            lap_loss = self.lambda_laplacian * self.laplacian_loss(depth_predicted, depth_gt)
        else:
            lap_loss = _zero

        ssim_loss_val = (self.ssim_loss(depth_predicted, depth_gt)
                         if self.ssim_loss is not None else _zero)

        if self.lambda_grad > 0:
            dx_pred = depth_predicted[:, :, :, 1:] - depth_predicted[:, :, :, :-1]
            dy_pred = depth_predicted[:, :, 1:, :] - depth_predicted[:, :, :-1, :]
            dx_gt = depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1]
            dy_gt = depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :]
            grad_loss_val = F.smooth_l1_loss(dx_pred, dx_gt) + F.smooth_l1_loss(dy_pred, dy_gt)
        else:
            grad_loss_val = _zero

        total_loss = (self.lambda_depth * depth_loss
                      + self.lambda_mat * mat_loss
                      + total_ct_loss + total_ccl_loss
                      + self.lambda_tv * tv_loss
                      + self.lambda_teacher_depth * teacher_depth_loss
                      + multiscale_depth_loss + lap_loss
                      + self.lambda_ssim * ssim_loss_val
                      + self.lambda_grad * grad_loss_val)

        if self.opt.mode == "train":
            if self._accum_step == 0:
                self.optimizer.zero_grad()

            if self._use_fp16:
                self.scaler.scale(total_loss / self._accum_steps).backward()
            else:
                (total_loss / self._accum_steps).backward()
            self._accum_step += 1

            if self._accum_step >= self._accum_steps:
                m = self._student_net()
                if self._use_fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(m.net_audio.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(m.ccl_audiomat.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(m.ccl_audiodepth.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(m.img_proj.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(m.aud_proj.parameters(), max_norm=1.0)
                if self._use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.warmup_scheduler is not None and self._warmup_step_count < self._warmup_steps:
                    self.warmup_scheduler.step()
                    self._warmup_step_count += 1
                elif self.scheduler is not None and not self._plateau_scheduler:
                    self.scheduler.step()
                # EMA update
                if self.ema_net is not None:
                    with torch.no_grad():
                        src = self._student_net().net_audio
                        for ema_p, src_p in zip(self.ema_net.parameters(), src.parameters()):
                            ema_p.mul_(self.ema_decay).add_(src_p, alpha=1 - self.ema_decay)
                        for ema_b, src_b in zip(self.ema_net.buffers(), src.buffers()):
                            ema_b.copy_(src_b)
                self._accum_step = 0

        losses = [total_loss.item(), depth_loss, mat_loss,
                  ccl_audiodepth_loss, ccl_audiomat_loss, loss_ct_ad, loss_ct_am,
                  teacher_depth_loss]
        return output, losses


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

opt = TrainOptions().parse()
opt.device = torch.device("cuda")

_seed = getattr(opt, 'seed', 0)
if _seed > 0:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if getattr(opt, 'detect_anomaly', False):
    torch.autograd.set_detect_anomaly(True)
    print("[WARN] Anomaly detection ON (slow).")

teacher_cache_path = opt.teacher_cache_path if opt.teacher_cache_path else None
validation_cache_path = opt.validation_cache_path if opt.validation_cache_path else None

writer = SummaryWriter(os.path.join("runs", opt.exp_name))
train_loss_file = TextWrite(os.path.join(opt.expr_dir, 'train_loss.csv'))
train_loss_file.add_line_csv(['step', 'loss'])
train_loss_file.write_line()

val_loss_file = TextWrite(os.path.join(opt.expr_dir, 'val_loss.csv'))
val_loss_file.add_line_csv(['step', 'loss'])
val_loss_file.write_line()

val_error_file = TextWrite(os.path.join(opt.expr_dir, 'val_error.csv'))
val_error_file.add_line_csv(['step', 'RMSE', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3'])
val_error_file.write_line()

ep_init = 0

# Build networks
builder = ModelBuilder()
if opt.resume_dir and opt.last_epoch:
    ep_init = opt.last_epoch
    _sfx = opt.dataset + '_epoch_' + str(ep_init)
    _ckpt = os.path.join(opt.resume_dir, opt.dataset)
    _backbone = getattr(opt, 'backbone', 'Resnet18')
    net_audiodepth = builder.build_audiodepth(
        backbone=_backbone, mode="mat", audio_shape=opt.audio_shape,
        weights=os.path.join(_ckpt, 'audiodepth_' + _sfx + '.pth'))
    net_rgbdepth = builder.build_rgbdepth(
        weights=os.path.join("checkpoints_pretrained", "rgbdepth_" + opt.dataset + ".pth"))
    net_material = builder.build_material_property(init_weights=opt.init_material_weight)
    freeze_weights(net_material)
    if opt.freeze_nets:
        freeze_weights(net_rgbdepth)
    ccl_audiodepth_net = builder.build_ccl(
        head="DepthNet", input_dim=512, normalization_sign=False,
        weights=os.path.join(_ckpt, 'ccl_audiodepth_' + _sfx + '.pth'))
    ccl_audiomat_net = builder.build_ccl(
        head="MatNet", input_dim=512, normalization_sign=False,
        weights=os.path.join(_ckpt, 'ccl_audiomat_' + _sfx + '.pth'))
    img_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128)
    aud_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128)
else:
    print("Training Compositional Contrastive Learning...")
    _backbone = getattr(opt, 'backbone', 'Resnet18')
    _init_aud_wt = getattr(opt, 'init_audiodepth_weight', '')
    net_audiodepth = builder.build_audiodepth(
        backbone=_backbone, mode="mat", audio_shape=opt.audio_shape, weights=_init_aud_wt)
    if getattr(opt, 'use_se_skips', False):
        net_audiodepth._use_se_skips = True
    img_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128)
    aud_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128)

    if teacher_cache_path:
        print("[INFO] Teacher cache enabled — teachers NOT loaded to GPU.")
        net_rgbdepth = builder.build_rgbdepth()
        net_material = builder.build_material_property(init_weights=opt.init_material_weight)
        net_rgbdepth.cpu()
        net_material.cpu()
        freeze_weights(net_rgbdepth)
        freeze_weights(net_material)
    else:
        net_rgbdepth = builder.build_rgbdepth(
            weights=os.path.join("checkpoints_pretrained", "rgbdepth_" + opt.dataset + ".pth"))
        net_material = builder.build_material_property(init_weights=opt.init_material_weight)
        freeze_weights(net_material)
        if opt.freeze_nets:
            freeze_weights(net_rgbdepth)

    ccl_audiomat_net = builder.build_ccl(head="MatNet", input_dim=512, normalization_sign=False)
    ccl_audiodepth_net = builder.build_ccl(head="DepthNet", input_dim=512, normalization_sign=False)

nets = (net_rgbdepth, net_audiodepth, net_material,
        ccl_audiomat_net, ccl_audiodepth_net, img_proj, aud_proj)

dataloader = CustomDatasetDataLoader()
dataloader.initialize(opt, teacher_cache_path=teacher_cache_path)
dataset = dataloader.load_data()
print(f'#training clips = {len(dataset)}')

if opt.validation_on:
    opt.mode = 'val'
    if validation_cache_path and not os.path.exists(validation_cache_path):
        print(f"[WARN] --validation_cache_path '{validation_cache_path}' not found")
        validation_cache_path = None
    dataloader_val = CustomDatasetDataLoader()
    dataloader_val.initialize(opt, teacher_cache_path=validation_cache_path)
    dataset_val = dataloader_val.load_data()
    print(f'#validation clips = {len(dataloader_val)}')
    opt.mode = 'train'

total_steps = 0
best_rmse = float("inf")
_mp3d_budget = getattr(opt, 'dataset', 'replica') in (
    'mp3d', 'mp3d_custom', 'mp3d_map', 'mp3d_map_odom_1m')
TRAIN_TIME_BUDGET = 3600 if _mp3d_budget else 300
_early_stop_patience = getattr(opt, 'early_stop_patience', 0)
_no_improve_count = 0

if getattr(opt, 'gradient_checkpointing', False):
    net_audiodepth.enable_gradient_checkpointing()

train = Trainer(nets, opt, augment=True)

_train_start = time.time()
_time_up = False
_early_stopped = False

for epoch in range(ep_init, ep_init + opt.niter):
    if _time_up or _early_stopped:
        break
    batch_loss = []
    for i, data in enumerate(dataset):
        if TRAIN_TIME_BUDGET:
            if time.time() - _train_start >= TRAIN_TIME_BUDGET:
                print(f"[INFO] Training budget reached at epoch {epoch}, step {i}.")
                _time_up = True
                break
        if _early_stopped:
            break

        total_steps += opt.batchSize
        output, losses = train.forward(data)
        step_loss, depth_loss, mat_loss, ccl_audiodepth_loss, ccl_audiomat_loss, \
            loss_ct_ad, loss_ct_am, teacher_depth_loss = losses
        batch_loss.append(step_loss)

        if total_steps // opt.batchSize % opt.display_freq == 0:
            print(f'[Epoch {epoch}, Step {total_steps // opt.batchSize}]')
            print(f"  depth_loss: {depth_loss:.4f}, mat_loss: {mat_loss}")
            print(f"  ccl_depth: {ccl_audiodepth_loss}, ccl_mat: {ccl_audiomat_loss}")
            print(f"  ct_ad: {loss_ct_ad}, ct_am: {loss_ct_am}")
            print(f"  teacher_depth: {teacher_depth_loss}")

            avg_loss = sum(batch_loss) / len(batch_loss)
            writer.add_images('train/audio_depth', output["audio_depth"], epoch, dataformats="NCHW")
            writer.add_images('train/depth_gt', output["depth_gt"], epoch, dataformats="NCHW")
            writer.add_scalar('train/avg_loss', avg_loss, epoch)
            writer.add_scalar('train/depth_loss', depth_loss, epoch)

            print(f'  avg_loss: {avg_loss:.5f}\n')
            batch_loss = []

        if total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on:
            train.model.eval()
            opt.mode = 'val'
            print(f'Validation at epoch {epoch}, step {total_steps // opt.batchSize}')
            with train.swap_ema():
                val_loss, val_err = evaluate(train, dataset_val, opt, writer, epoch)
            writer.add_scalar('val/Loss', val_loss, epoch)
            writer.add_scalar('val/RMSE', val_err["RMSE/STD"], epoch)

            train.model.train()
            opt.mode = 'train'

            if train._plateau_scheduler and train.scheduler is not None:
                _prev_lr = train.optimizer.param_groups[0]['lr']
                train.scheduler.step(val_err['RMSE/STD'])
                _new_lr = train.optimizer.param_groups[0]['lr']
                if _new_lr < _prev_lr:
                    print(f'[LR] {_prev_lr:.2e} -> {_new_lr:.2e}')

            if val_err['RMSE/STD'] < best_rmse:
                best_rmse = val_err['RMSE/STD']
                _no_improve_count = 0
                print(f'Best model (epoch {epoch}) RMSE: {val_err["RMSE/STD"]:.5f}\n')
            else:
                _no_improve_count += 1
                if _early_stop_patience > 0 and _no_improve_count >= _early_stop_patience:
                    print(f'[INFO] Early stopping (no improvement for '
                          f'{_no_improve_count} checks). Best RMSE: {best_rmse:.5f}')
                    _early_stopped = True
                torch.save(net_audiodepth.state_dict(),
                           os.path.join(opt.expr_dir, f'audiodepth_{opt.dataset}.pth'))
                torch.save(ccl_audiodepth_net.state_dict(),
                           os.path.join(opt.expr_dir, f'ccl_audiodepth_{opt.dataset}.pth'))
                torch.save(ccl_audiomat_net.state_dict(),
                           os.path.join(opt.expr_dir, f'ccl_audiomat_{opt.dataset}.pth'))

    if epoch % opt.epoch_save_freq == 0:
        print(f'Saving model at epoch {epoch}')
        torch.save(net_audiodepth.state_dict(),
                   os.path.join(opt.expr_dir, f'audiodepth_{opt.dataset}_epoch_{epoch}.pth'))
        torch.save(ccl_audiodepth_net.state_dict(),
                   os.path.join(opt.expr_dir, f'ccl_audiodepth_{opt.dataset}_epoch_{epoch}.pth'))
        torch.save(ccl_audiomat_net.state_dict(),
                   os.path.join(opt.expr_dir, f'ccl_audiomat_{opt.dataset}_epoch_{epoch}.pth'))

    if opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0:
        train.decrease_learning_rate(opt.decay_factor)

# Final validation
if opt.validation_on:
    train.model.eval()
    opt.mode = 'val'
    print('Final validation:')
    with train.swap_ema():
        val_loss, val_err = evaluate(train, dataset_val, opt, writer, epoch)
    writer.add_scalar('val/Loss', val_loss, epoch)
    train.model.train()
    opt.mode = 'train'

writer.close()

import contextlib
import copy
import gc
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from models import criterion
from models.audioVisual_model_ccl import AudioVisualModel
from models.models import ModelBuilder
from options.train_options import TrainOptions
from util.util import compute_errors, AverageMeter


def freeze_weights(net):
    for p in net.parameters():
        p.requires_grad = False


def cleanup_old_checkpoints(expr_dir, dataset, keep_last=1):
    for prefix in ('audiodepth', 'ccl_audiodepth', 'ccl_audiomat'):
        files = sorted(glob.glob(os.path.join(expr_dir, f'{prefix}_{dataset}_epoch_*.pth')),
                       key=lambda p: int(p.rsplit('_epoch_', 1)[-1][:-4]))
        for f in files[:-keep_last]:
            os.remove(f)


def info_nce(s, t, tau):
    logits = s @ t.T / tau
    y = torch.arange(s.shape[0], device=s.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def evaluate(trainer, loader, opt, writer, step, max_batches=None):
    losses, errors = [], []
    depth_losses, mat_losses = AverageMeter(), AverageMeter()
    ccl_ad, ccl_am = AverageMeter(), AverageMeter()
    median_scale = 'batvision' in opt.dataset
    with torch.no_grad():
        for i, data in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            output, l = trainer.forward(data)
            _, depth_loss, mat_loss, l_ccl_ad, l_ccl_am = l[:5]
            depth_losses.update(depth_loss, opt.batchSize)
            mat_losses.update(mat_loss, opt.batchSize)
            ccl_ad.update(l_ccl_ad, opt.batchSize)
            ccl_am.update(l_ccl_am, opt.batchSize)
            losses.append(trainer.lambda_depth * depth_loss + trainer.lambda_mat * mat_loss)
            gt = output['depth_gt'].float().cpu().numpy()
            pred = output['audio_depth'].float().cpu().numpy()
            for b in range(pred.shape[0]):
                mask = (gt[b] > 0) & (gt[b] < opt.max_depth)
                errors.append(compute_errors(gt[b], pred[b], mask=mask, median_scale=median_scale))

    mean_loss = sum(losses) / len(losses)
    e = np.array(errors).mean(0)
    print('Loss: {:.3f}, RMSE: {:.3f}, delta1: {:.3f}, delta2: {:.3f}, delta3: {:.3f}'.format(
        mean_loss, e[1], e[2], e[3], e[4]))
    val_errors = {'ABS_REL': e[0], 'RMSE': e[1], 'DELTA1': e[2], 'DELTA2': e[3], 'DELTA3': e[4]}

    n = opt.val_split_n
    if n > 0 and len(errors) > n:
        ev, et = np.array(errors[:n]).mean(0), np.array(errors[n:]).mean(0)
        print('[split] val n={} abs_rel {:.4f} rmse {:.4f} d1 {:.4f} | test n={} abs_rel {:.4f} rmse {:.4f} d1 {:.4f}'.format(
            n, ev[0], ev[1], ev[2], len(errors) - n, et[0], et[1], et[2]))
        val_errors['RMSE_VALONLY'] = ev[1]
        writer.add_scalar('val_only/RMSE', ev[1], step)
        writer.add_scalar('val_only/ABS_REL', ev[0], step)
        writer.add_scalar('test_only/RMSE', et[1], step)
        writer.add_scalar('test_only/ABS_REL', et[0], step)

    writer.add_scalar('val/mat_loss', mat_losses.avg, step)
    writer.add_scalar('val/depth_loss', depth_losses.avg, step)
    writer.add_scalar('val/ccl_audiomat_loss', ccl_am.avg, step)
    writer.add_scalar('val/ccl_audiodepth_loss', ccl_ad.avg, step)
    return mean_loss, val_errors


class Trainer:
    def __init__(self, nets, opt):
        self.opt = opt
        self.model = AudioVisualModel(nets, opt).to(opt.device)
        m = self.model

        self.lambda_depth = opt.lambda_depth
        self.lambda_mat = opt.lambda_mat
        self.lambda_ccl_depth = opt.lambda_ccl_depth
        self.lambda_ccl_mat = opt.lambda_ccl_mat
        self.lambda_ct = opt.lambda_ct
        self.lambda_teacher_depth = opt.lambda_teacher_depth
        self.lambda_ldc = opt.lambda_ldc
        self.lambda_cc = opt.lambda_cc
        self.lambda_multiscale = opt.lambda_multiscale
        self.lambda_laplacian = opt.lambda_laplacian
        self.lambda_ssim = opt.lambda_ssim
        self.lambda_grad = opt.lambda_grad
        self.lambda_feat_std = opt.lambda_feat_std
        if opt.lambda_depth == 0 and opt.mode == 'train':
            raise ValueError('--lambda_depth 0 leaves the depth head unsupervised')

        md = opt.max_depth
        self.loss_criterion = {
            'silog': lambda: criterion.SILogLoss(scaling_factor=1.0, max_depth=md),
            'l1': criterion.L1Loss,
            'l2': criterion.L2Loss,
            'berhu': lambda: criterion.Berhuloss(threshold=opt.berhu_threshold, max_depth=md),
        }.get(opt.depth_loss_type, lambda: criterion.LogDepthLoss(max_depth=md))()
        self.laplacian_loss = criterion.LaplacianDepthLoss(max_depth=md) if self.lambda_laplacian > 0 else None
        self.ssim_loss = criterion.SSIMLoss() if self.lambda_ssim > 0 else None
        self.feat_std_hinge = criterion.FeatureStdHinge(gamma=opt.feat_std_gamma) if self.lambda_feat_std > 0 else None
        self.mat_criterion = criterion.DistillationLoss()
        if opt.ccl_depth_loss == 'berhu':
            self.ccl_audiodepth_criterion = criterion.Berhuloss(threshold=0.2, max_depth=md)
        else:
            self.ccl_audiodepth_criterion = criterion.LogDepthLoss(zero_depth_weight=0.9, max_depth=md)
        self.ct_am_criterion = criterion.CosineCCLLoss(temperature=opt.ccl_temperature)
        if opt.ct_ad_loss == 'infonce':
            self.ct_ad_criterion = criterion.ContrastiveLoss(temperature=opt.ct_temperature)
        else:
            self.ct_ad_criterion = criterion.CosineCCLLoss(temperature=opt.ccl_temperature)

        backbone = list(m.net_audio.feature_extraction.parameters()) \
            if hasattr(m.net_audio, 'feature_extraction') else []
        ids = set(id(p) for p in backbone)
        other = [p for p in m.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{'params': backbone, 'lr': opt.lr_backbone_main, 'name': 'audio_backbone'},
                  {'params': other, 'lr': opt.lr_audio, 'name': 'other'}]
        if opt.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=opt.cosine_T_max, eta_min=1e-6) if opt.cosine_T_max > 0 else None
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=opt.warmup_steps) if opt.warmup_steps > 0 else None
        self._warmup_count = 0
        self._accum_steps = max(1, opt.accumulation_steps)
        self._accum_step = 0

        self.ema_decay = opt.ema_decay
        self.ema_net = None
        if self.ema_decay > 0:
            self.ema_net = copy.deepcopy(m.net_audio)
            for p in self.ema_net.parameters():
                p.requires_grad_(False)

        if opt.teacher_cache_path:
            if opt.rgb_teacher != 'moge_v2':
                m.net_rgbdepth.cpu()
            elif opt.cache_freeze_teacher_proj and opt.freeze_nets:
                freeze_weights(m.net_rgbdepth)
            m.net_material.cpu()
        torch.cuda.empty_cache()
        self._zero = torch.zeros(1, device=opt.device)
        self.last = {}

    @contextlib.contextmanager
    def swap_ema(self):
        if self.ema_net is None:
            yield
            return
        net = self.model.net_audio
        live = copy.deepcopy(net.state_dict())
        net.load_state_dict(self.ema_net.state_dict())
        try:
            yield
        finally:
            net.load_state_dict(live)

    def decrease_learning_rate(self, decay_factor=0.94):
        for g in self.optimizer.param_groups:
            g['lr'] *= decay_factor

    def forward(self, data):
        opt, zero = self.opt, self._zero
        data = {k: v.to(opt.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
        compute_ccl = max(self.lambda_ccl_depth, self.lambda_ccl_mat) != 0
        compute_ct = self.lambda_ct != 0
        output = self.model(data, compute_ccl=compute_ccl, compute_ct=compute_ct)

        gt = output['depth_gt']
        pred = output['audio_depth']
        depth_loss = self.loss_criterion(pred, gt)
        mat_loss = self.mat_criterion([output['audio_mat_class'], output['material_class']], None) \
            if self.lambda_mat != 0 and output['audio_mat_class'] is not None else zero

        img_depth = output['img_depth']
        td_loss = zero
        if self.lambda_teacher_depth > 0 and img_depth is not None:
            if img_depth.shape[-2:] != pred.shape[-2:]:
                img_depth = F.interpolate(img_depth.float(), size=pred.shape[-2:], mode='bilinear', align_corners=False)
            valid = (gt > 0) & (gt < opt.max_depth)
            if valid.any() and opt.teacher_depth_align:
                t = img_depth.detach().float().clone()
                for b in range(t.shape[0]):
                    mb = valid[b] & (t[b] > 1e-3)
                    if mb.sum() >= 50:
                        t[b] = t[b] * (gt[b][mb].median() / t[b][mb].median().clamp_min(1e-3))
                t = torch.where(valid, t.clamp(0, opt.max_depth), torch.zeros_like(t))
                td_loss = self.loss_criterion(pred, t)
            elif valid.any():
                if opt.berhu_teacher:
                    td_loss = self.loss_criterion(pred * valid.float(), img_depth.detach() * valid.float())
                else:
                    td_loss = F.smooth_l1_loss(pred[valid], img_depth.detach()[valid])

        if compute_ccl:
            ccl_ad_loss = self.ccl_audiodepth_criterion(output['ccl_audiodepth'], gt) \
                if output['ccl_audiodepth'] is not None else zero
            ccl_am_loss = self.mat_criterion([output['ccl_audiomat'], output['material_class']], None)
        else:
            ccl_ad_loss = ccl_am_loss = zero

        if compute_ct:
            ct_ad_loss = self.ct_ad_criterion(output['aud_proj_feat'], output['img_proj_feat'])
            ct_am_loss = opt.ct_am_scale * self.ct_am_criterion(
                output['audio_feat'].mean((2, 3)), output['material_feat'].detach().mean((2, 3)))
        else:
            ct_ad_loss = ct_am_loss = zero

        ldc_loss = zero
        if self.lambda_ldc > 0 and output['ldc_emb'] is not None and output['img_depth'] is not None \
                and output['ldc_emb'].shape[0] > 2:
            with torch.no_grad():
                lt = torch.log(output['img_depth'].float().clamp(0.1, opt.max_depth))
                lt = F.adaptive_avg_pool2d(lt, opt.ldc_grid).flatten(1)
                lt = F.normalize(lt - lt.mean(0, keepdim=True), dim=1)
            ldc_loss = info_nce(F.normalize(output['ldc_emb'].float(), dim=1), lt, opt.ldc_tau)

        cc_loss = zero
        if self.lambda_cc > 0 and output['cc_emb'] is not None and output['ccl_audiodepth_feat'] is not None \
                and output['cc_emb'].shape[0] > 2:
            with torch.no_grad():
                ct = F.adaptive_avg_pool2d(output['ccl_audiodepth_feat'].detach().float(), 4).flatten(1)
                ct = F.normalize(ct - ct.mean(0, keepdim=True), dim=1)
            cc_loss = info_nce(F.normalize(output['cc_emb'].float(), dim=1), ct, opt.cc_tau)

        total_ccl = self.lambda_ccl_depth * ccl_ad_loss + self.lambda_ccl_mat * ccl_am_loss
        total_ct = self.lambda_ct * (ct_ad_loss + ct_am_loss)

        ms_loss = zero
        if self.lambda_multiscale > 0:
            for s in (2, 4):
                ms_loss = ms_loss + self.loss_criterion(F.avg_pool2d(pred, s, stride=s), F.avg_pool2d(gt, s, stride=s))
            ms_loss = self.lambda_multiscale * ms_loss
        lap_loss = self.lambda_laplacian * self.laplacian_loss(pred, gt) if self.laplacian_loss is not None else zero
        ssim_loss = self.ssim_loss(pred, gt) if self.ssim_loss is not None else zero
        grad_loss = zero
        if self.lambda_grad > 0:
            grad_loss = (F.smooth_l1_loss(pred[..., 1:] - pred[..., :-1], gt[..., 1:] - gt[..., :-1])
                         + F.smooth_l1_loss(pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]))
        feat_std_loss = self.feat_std_hinge(output['audio_feat']) if self.feat_std_hinge is not None else zero

        total_loss = (self.lambda_feat_std * feat_std_loss
                      + self.lambda_depth * depth_loss
                      + self.lambda_mat * mat_loss
                      + total_ct + total_ccl
                      + self.lambda_ldc * ldc_loss
                      + self.lambda_cc * cc_loss
                      + self.lambda_teacher_depth * td_loss
                      + ms_loss + lap_loss
                      + self.lambda_ssim * ssim_loss
                      + self.lambda_grad * grad_loss)

        self.last = {'ldc': float(ldc_loss), 'cc': float(cc_loss), 'feat_std': float(feat_std_loss)}
        if opt.mode == 'train':
            m = self.model
            if self._accum_step == 0:
                self.optimizer.zero_grad()
            (total_loss / self._accum_steps).backward()
            self._accum_step += 1
            if self._accum_step < self._accum_steps:
                return output, self._losses(total_loss, depth_loss, mat_loss, ccl_ad_loss, ccl_am_loss,
                                            ct_ad_loss, ct_am_loss, td_loss)
            self._accum_step = 0
            for net in (m.net_audio, m.ccl_audiomat, m.ccl_audiodepth, m.img_proj, m.aud_proj):
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.warmup_scheduler is not None and self._warmup_count < opt.warmup_steps:
                self.warmup_scheduler.step()
                self._warmup_count += 1
            elif self.scheduler is not None:
                self.scheduler.step()
            if self.ema_net is not None:
                with torch.no_grad():
                    for e, s in zip(self.ema_net.parameters(), m.net_audio.parameters()):
                        e.mul_(self.ema_decay).add_(s, alpha=1 - self.ema_decay)
                    for e, s in zip(self.ema_net.buffers(), m.net_audio.buffers()):
                        e.copy_(s)

        return output, self._losses(total_loss, depth_loss, mat_loss, ccl_ad_loss, ccl_am_loss,
                                    ct_ad_loss, ct_am_loss, td_loss)

    @staticmethod
    def _losses(*terms):
        return [t.item() for t in terms]


def save_nets(opt, suffix, audio_net, ccl_ad, ccl_am):
    torch.save(audio_net.state_dict(), os.path.join(opt.expr_dir, f'audiodepth_{opt.dataset}{suffix}.pth'))
    torch.save(ccl_ad.state_dict(), os.path.join(opt.expr_dir, f'ccl_audiodepth_{opt.dataset}{suffix}.pth'))
    torch.save(ccl_am.state_dict(), os.path.join(opt.expr_dir, f'ccl_audiomat_{opt.dataset}{suffix}.pth'))


opt = TrainOptions().parse()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('[WARN] --seed 0: run is unseeded')
if opt.deterministic:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)

teacher_cache_path = opt.teacher_cache_path or None
validation_cache_path = opt.validation_cache_path or None
material_cache_path = opt.material_cache_path or None
if material_cache_path and not teacher_cache_path:
    raise ValueError('--material_cache_path needs --teacher_cache_path')

writer = SummaryWriter(os.path.join('runs', opt.exp_name))
builder = ModelBuilder()
audiodepth_kwargs = dict(
    use_se_skips=opt.use_se_skips,
    decoder_spatial_entry=opt.decoder_spatial_entry,
    use_silu=opt.use_silu,
    use_residual_decoder=opt.use_residual_decoder,
    wider_decoder=opt.wider_decoder,
    use_log_depth=opt.use_log_depth,
    max_depth=float(opt.max_depth),
    decoder_dropout=opt.audio_decoder_dropout if opt.audio_decoder_dropout >= 0 else 0.1,
    target_hw=tuple(int(x) for x in opt.depth_target_hw.split(',')),
    audio_norm_type=opt.audio_norm_type,
    audio_norm_groups=opt.audio_norm_groups,
    mat_nclass=opt.mat_nclass,
)
rgbdepth_kwargs = dict(
    teacher=opt.rgb_teacher,
    teacher_max_depth=float(opt.max_depth),
    moge_model_id=opt.moge_model_id,
    moge_num_tokens=opt.moge_num_tokens or None,
    moge_resolution_level=opt.moge_resolution_level,
)
rgb_weights = os.path.join('checkpoints_pretrained', f'rgbdepth_{opt.dataset}.pth')

ep_init = 0
if opt.resume_dir and opt.last_epoch is not None:
    ep_init = opt.last_epoch
    ckpt = os.path.join(opt.resume_dir, opt.dataset)
    sfx = f'{opt.dataset}_epoch_{ep_init}.pth'
    net_audiodepth = builder.build_audiodepth(
        backbone=opt.backbone, mode='mat', audio_shape=opt.audio_shape,
        weights=os.path.join(ckpt, 'audiodepth_' + sfx), **audiodepth_kwargs)
    net_rgbdepth = builder.build_rgbdepth(weights=rgb_weights, **rgbdepth_kwargs)
    net_material = builder.build_material_property(init_weights=opt.init_material_weight)
    freeze_weights(net_material)
    if opt.freeze_nets:
        freeze_weights(net_rgbdepth)
    ccl_audiodepth_net = builder.build_ccl(head='DepthNet', input_dim=512, use_film=opt.use_film_ce,
                                           weights=os.path.join(ckpt, 'ccl_audiodepth_' + sfx))
    ccl_audiomat_net = builder.build_ccl(head='MatNet', input_dim=512, use_film=opt.use_film_ce,
                                         weights=os.path.join(ckpt, 'ccl_audiomat_' + sfx))
    img_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128, use_ln=opt.use_ln_proj)
    aud_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128, use_ln=opt.use_ln_proj)
else:
    net_audiodepth = builder.build_audiodepth(
        backbone=opt.backbone, mode='mat', audio_shape=opt.audio_shape,
        weights=opt.init_audiodepth_weight, **audiodepth_kwargs)
    img_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128, use_ln=opt.use_ln_proj)
    aud_proj = builder.build_latent_proj_head(in_fc=512, out_fc=128, use_ln=opt.use_ln_proj)
    if teacher_cache_path and opt.rgb_teacher == 'moge_v2':
        net_rgbdepth = builder.build_rgbdepth(moge_cache_consumer_mode=True,
                                              moge_cache_enc_dim_out=opt.moge_cache_enc_dim_out,
                                              **rgbdepth_kwargs)
    elif teacher_cache_path:
        net_rgbdepth = builder.build_rgbdepth(**rgbdepth_kwargs)
        freeze_weights(net_rgbdepth)
    else:
        net_rgbdepth = builder.build_rgbdepth(weights=rgb_weights, **rgbdepth_kwargs)
    net_material = builder.build_material_property(init_weights=opt.init_material_weight)
    freeze_weights(net_material)
    if not teacher_cache_path and opt.freeze_nets:
        freeze_weights(net_rgbdepth)
    ccl_audiomat_net = builder.build_ccl(head='MatNet', input_dim=512, use_film=opt.use_film_ce,
                                         n_class=opt.mat_nclass)
    ccl_audiodepth_net = builder.build_ccl(head='DepthNet', input_dim=512, use_film=opt.use_film_ce)

nets = (net_rgbdepth, net_audiodepth, net_material, ccl_audiomat_net, ccl_audiodepth_net, img_proj, aud_proj)

dataloader = CustomDatasetDataLoader()
dataloader.initialize(opt, teacher_cache_path=teacher_cache_path, material_cache_path=material_cache_path)
dataset = dataloader.load_data()
print(f'#training clips = {len(dataset)}')

if opt.validation_on:
    opt.mode = 'val'
    if validation_cache_path and not os.path.exists(validation_cache_path):
        print(f'[WARN] --validation_cache_path {validation_cache_path} not found')
        validation_cache_path = None
    n_threads, opt.nThreads = opt.nThreads, 0
    dataloader_val = CustomDatasetDataLoader()
    dataloader_val.initialize(opt, teacher_cache_path=validation_cache_path,
                              material_cache_path=opt.validation_material_cache_path or None
                              if validation_cache_path else None)
    dataset_val = dataloader_val.load_data()
    print(f'#validation clips = {len(dataloader_val)}')
    if hasattr(getattr(dataset_val, 'dataset', None), '_orientations'):
        from data_loader.audio_visual_dataset import preload_audio_for_dataset
        preload_audio_for_dataset(dataset_val.dataset)
    opt.nThreads = n_threads
    opt.mode = 'train'

train = Trainer(nets, opt)
total_steps = 0
best_rmse = best_rmse_valonly = float('inf')
no_improve = 0
early_stopped = False

for epoch in range(ep_init, ep_init + opt.niter):
    if early_stopped:
        break
    batch_loss = []
    for i, data in enumerate(dataset):
        if early_stopped:
            break
        total_steps += opt.batchSize
        step = total_steps // opt.batchSize
        output, losses = train.forward(data)
        del data, output
        step_loss, depth_loss, mat_loss, ccl_ad_loss, ccl_am_loss, ct_ad_loss, ct_am_loss, td_loss = losses
        batch_loss.append(step_loss)
        if step % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if step % opt.display_freq == 0:
            print(f'[Epoch {epoch}, Step {step}]')
            print(f'  depth_loss: {depth_loss:.4f}, mat_loss: {mat_loss:.4f}')
            print(f'  ccl_depth: {ccl_ad_loss:.4f}, ccl_mat: {ccl_am_loss:.4f}')
            print(f'  ct_ad: {ct_ad_loss:.4f}, ct_am: {ct_am_loss:.4f}')
            print(f'  teacher_depth: {td_loss:.4f}')
            if train.lambda_ldc > 0:
                print(f"  ldc: {train.last['ldc']:.4f}")
            if train.lambda_cc > 0:
                print(f"  cc: {train.last['cc']:.4f}")
            avg_loss = sum(batch_loss) / len(batch_loss)
            for k, v in (('avg_loss', avg_loss), ('depth_loss', depth_loss), ('mat_loss', mat_loss),
                         ('ccl_audiodepth_loss', ccl_ad_loss), ('ccl_audiomat_loss', ccl_am_loss),
                         ('teacher_depth_loss', td_loss), ('feat_std_loss', train.last['feat_std']),
                         ('lr', train.optimizer.param_groups[0]['lr'])):
                writer.add_scalar(f'train/{k}', v, step)
            writer.flush()
            print(f'  avg_loss: {avg_loss:.5f}\n')
            batch_loss = []

        if opt.validation_on and step % opt.validation_freq == 0:
            train.model.eval()
            opt.mode = 'val'
            print(f'Validation at epoch {epoch}, step {step}')
            with train.swap_ema():
                val_loss, val_err = evaluate(train, dataset_val, opt, writer, step,
                                             max_batches=opt.val_max_batches or None)
            writer.add_scalar('val/Loss', val_loss, step)
            for k in ('RMSE', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3'):
                writer.add_scalar(f'val/{k}', val_err[k], step)
            writer.flush()
            train.model.train()
            opt.mode = 'train'

            best_net = train.ema_net if train.ema_net is not None else net_audiodepth
            if val_err['RMSE'] < best_rmse:
                best_rmse = val_err['RMSE']
                no_improve = 0
                print(f'Best model (epoch {epoch}) RMSE: {best_rmse:.5f}\n')
                save_nets(opt, '', best_net, ccl_audiodepth_net, ccl_audiomat_net)
            else:
                no_improve += 1
                if 0 < opt.early_stop_patience <= no_improve:
                    print(f'[INFO] Early stopping (no improvement for {no_improve} checks). Best RMSE: {best_rmse:.5f}')
                    early_stopped = True
            vo = val_err.get('RMSE_VALONLY')
            if vo is not None and vo < best_rmse_valonly:
                best_rmse_valonly = vo
                print(f'Best val-only model (epoch {epoch}) RMSE: {vo:.5f}')
                save_nets(opt, '_bestval', best_net, ccl_audiodepth_net, ccl_audiomat_net)

    if epoch % opt.epoch_save_freq == 0:
        print(f'Saving model at epoch {epoch}')
        save_nets(opt, f'_epoch_{epoch}', net_audiodepth, ccl_audiodepth_net, ccl_audiomat_net)
        cleanup_old_checkpoints(opt.expr_dir, opt.dataset)
    if opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0:
        train.decrease_learning_rate(opt.decay_factor)

if opt.val_split_n > 0:
    print(f'Saving last model (epoch {epoch})')
    save_nets(opt, '_last', train.ema_net if train.ema_net is not None else net_audiodepth,
              ccl_audiodepth_net, ccl_audiomat_net)

if opt.validation_on:
    train.model.eval()
    opt.mode = 'val'
    print('Final validation:')
    with train.swap_ema():
        val_loss, val_err = evaluate(train, dataset_val, opt, writer, total_steps // opt.batchSize)
    writer.add_scalar('val/Loss', val_loss, total_steps // opt.batchSize)
    train.model.train()
    opt.mode = 'train'
writer.close()

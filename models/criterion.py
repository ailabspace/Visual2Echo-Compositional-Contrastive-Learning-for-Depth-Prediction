import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import SSIM


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)
            if len(preds) == 2:
                err = self._forward([preds[0], preds[1]], targets, weight)
            else:
                errs = [self._forward(preds[n], targets[n], weight[n])
                        for n in range(N)]
                err = torch.mean(torch.stack(errs))
        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds[0].new_ones(1)
            err = self._forward(preds, targets, weight)
        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class LogDepthLoss(BaseLoss):
    def __init__(self, zero_depth_weight=0.1, max_depth=float('inf')):
        super(LogDepthLoss, self).__init__()
        self.zero_depth_weight = zero_depth_weight
        self.max_depth = max_depth

    def _forward(self, pred, target, weight):
        assert pred.shape == target.shape, (
            f"LogDepthLoss: pred {tuple(pred.shape)} != target {tuple(target.shape)}. "
            "Interpolate prediction to GT size before calling."
        )
        valid = (target > 0) & (target < self.max_depth)
        log_loss = torch.log(torch.abs(pred - target) + 1)
        w = torch.where(valid, torch.ones_like(target),
            torch.where(target <= 0, torch.full_like(target, self.zero_depth_weight),
                        torch.zeros_like(target)))
        if weight is not None:
            w = w * weight
        return (log_loss * w).sum() / w.sum().clamp(min=1.0)


class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, pred, target):
        return F.mse_loss(pred, target)


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target)


class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.2)

    def _forward(self, pred, target, weight):
        return self.loss(pred, target)


class DepthAwareLoss(BaseLoss):
    def __init__(self, alpha=0.5, zero_depth_weight=0.1, max_depth=10):
        super(DepthAwareLoss, self).__init__()
        self.alpha = alpha
        self.zero_depth_weight = zero_depth_weight
        self.max_depth = max_depth

    def _forward(self, pred, target, weight):
        non_zeros = target > 0
        depth_weight = torch.where(non_zeros, 1. / (target + self.alpha), self.zero_depth_weight)
        loss = F.mse_loss(pred, target, reduction='none')
        weighted_loss = depth_weight * loss
        return weighted_loss.mean()


class KnowledgeDistillationLoss(BaseLoss):
    """Combines hard-target log loss with soft-target KL divergence."""

    def __init__(self, alpha=0.5, temperature=1.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def _forward(self, pred, target, weight):
        student_pred, teacher_pred = pred[0], pred[1]
        log_loss = torch.mean(torch.log(torch.abs(
            student_pred[target != 0] - target[target != 0]) + 1))

        soft_target = F.softmax(teacher_pred / self.temperature, dim=1)
        soft_pred = F.log_softmax(student_pred / self.temperature, dim=1)
        kd_loss = F.kl_div(soft_pred, soft_target, reduction='batchmean')

        loss = (1.0 - self.alpha) * log_loss + self.alpha * kd_loss
        return loss


class LaplacianDepthLoss(nn.Module):
    """Edge-aware depth loss using Laplacian filter."""

    def __init__(self, max_depth=10.0):
        super(LaplacianDepthLoss, self).__init__()
        self.max_depth = max_depth
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel', laplacian_kernel)

    def forward(self, pred, gt):
        mask = (gt > 0) & (gt < self.max_depth)
        kernel = self.kernel.to(pred.device)
        pred_lap = F.conv2d(pred, kernel, padding=1)
        gt_lap = F.conv2d(gt, kernel, padding=1)
        mask_exp = mask.expand_as(pred_lap)
        if mask_exp.sum() == 0:
            return pred.new_zeros(1)
        return F.smooth_l1_loss(pred_lap[mask_exp], gt_lap[mask_exp])


class DistillationLoss(BaseLoss):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def _forward(self, pred, targets, weight):
        student_pred, teacher_pred = pred[0], pred[1]
        student_pred = torch.clamp(student_pred, -50., 50.)
        teacher_pred = torch.clamp(teacher_pred, -50., 50.)
        student_soft = F.log_softmax(student_pred / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_pred / self.temperature, dim=1)

        distillation_loss = self.kl_div_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        return distillation_loss


class NCELoss(BaseLoss):
    """Noise-contrastive estimation loss for cross-modal alignment."""

    def __init__(self, temperature=1., head=None):
        super(NCELoss, self).__init__()
        assert head is not None
        self.head = head
        self.temperature = temperature
        self._epsilon = 1e-6

    def _flatten_norm(self, feat):
        batch_size = feat.size(0)
        f_flat = feat.reshape(batch_size, -1)
        norms = torch.norm(f_flat, p=2, dim=1, keepdim=True)
        return f_flat / (norms + 1e-10)

    def forward(self, f1, f2, targets):
        f1 = self._flatten_norm(f1)
        f2 = self._flatten_norm(f2)

        if self.head == "MatNet":
            if len(f2.size()) > 2 and len(f1.size()) > 2:
                f2 = f2.view(f1.size(0), -1)
                f1 = f1.view(f2.size(0), -1)
            else:
                f1 = f1.view(f2.size(0), -1)

            sim_matrix = torch.mm(f1, f2.T) / self.temperature

            targets = targets.unsqueeze(1)
            mask = torch.eq(targets, targets.transpose(0, 1)).float().sum(dim=2)
            mask.fill_diagonal_(1)

            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            logits = sim_matrix - logits_max.detach()

            exp_logits = torch.exp(logits)
            exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

            log_prob = logits - torch.log(exp_logits_sum + self._epsilon)

            mask = mask / mask.sum(dim=1, keepdim=True)
            loss = -(mask * log_prob).sum(dim=1).mean()

        elif self.head == "DepthNet":
            labels = torch.eye(f1.size(0), device=f1.device)
            mask = 1 - labels
            sim_matrix = torch.mm(f1, f2.T) / self.temperature
            sim_matrix_scaled = sim_matrix * mask

            logits_max, _ = torch.max(sim_matrix_scaled, dim=1, keepdim=True)
            logits = sim_matrix_scaled - logits_max.detach()
            exp_logits = torch.exp(logits)
            exp_logits_sum = exp_logits.sum(dim=1, keepdim=True) + self._epsilon

            log_prob = logits - torch.log(exp_logits_sum)
            loss = -(labels * log_prob).sum(dim=1).mean()

        return loss


class ContrastiveLoss(BaseLoss):
    """Symmetric contrastive loss (InfoNCE-style)."""

    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f1, f2):
        batch_sz = f1.size(0)
        f1 = f1.view(batch_sz, -1)
        f2 = f2.view(batch_sz, -1)
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

        sim_matrix = torch.mm(f1, f2.T) / self.temperature

        labels = torch.arange(batch_sz, device=f1.device)
        loss_f1_to_f2 = F.cross_entropy(sim_matrix, labels)
        loss_f2_to_f1 = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_f1_to_f2 + loss_f2_to_f1) / 2.
        return loss


class CosineCCLLoss(nn.Module):
    """Symmetric cosine similarity loss for cross-modal feature alignment (Eq. 13).

    L_CCL = [1 - sim(f^t, f^s)] + [1 - sim(f^s, f^t)]
    where sim(a, b) = a . b / (||a|| ||b||).
    """

    def __init__(self, temperature=1.0):
        super(CosineCCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_t, f_s):
        f_t = f_t.reshape(f_t.size(0), -1)
        f_s = f_s.reshape(f_s.size(0), -1)
        f_t_n = F.normalize(f_t, p=2, dim=1)
        f_s_n = F.normalize(f_s, p=2, dim=1)
        sim = (f_t_n * f_s_n).sum(dim=1) / self.temperature
        loss = 2.0 * (1.0 - sim).mean()
        return loss


class SILogLoss(BaseLoss):
    """Scale-invariant logarithmic loss for depth estimation."""

    def __init__(self, variance_focus=0.85, scaling_factor=10., max_depth=float('inf')):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.scaling_factor = scaling_factor
        self.max_depth = max_depth

    def forward(self, pred, target):
        valid = (target > 0) & (pred > 0) & (target < self.max_depth)
        d = torch.log(pred[valid]) - torch.log(target[valid])
        return torch.sqrt(
            (d ** 2).mean() - self.variance_focus * (d.mean() ** 2) + 1e-6
        ) * self.scaling_factor


class Berhuloss(BaseLoss):
    """Reverse Huber (BerHu) loss for depth estimation."""

    def __init__(self, threshold=0.2, max_depth=float('inf')):
        super(Berhuloss, self).__init__()
        self.threshold = threshold
        self.max_depth = max_depth

    def forward(self, pred, target, mask=None, d_map=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid = (target > 0) & (target < self.max_depth)
        if not valid.any():
            return pred.sum() * 0.0
        pred_v = pred[valid]
        target_v = target[valid]

        diff = torch.abs(target_v - pred_v)
        delta = self.threshold * torch.max(diff).item()

        p1 = -F.threshold(-diff, -delta, 0.)
        p2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
        p2 = p2 / (2. * delta)
        diff = p1 + p2
        return diff.mean()


class ReprojectionLoss(BaseLoss):
    def __init__(self, no_ssim=False):
        super(ReprojectionLoss, self).__init__()
        self.no_ssim = no_ssim
        if not self.no_ssim:
            self.ssim = SSIM()

    def forward(self, pred, target, sigma):
        abs_diff = torch.log(torch.abs(target - pred) + 1).mean(1, keepdim=True)
        l1_loss = abs_diff

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            self.ssim.to(pred.device)
            ssim_loss = self.ssim(pred, target)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
            reprojection_loss = reprojection_loss * sigma

        return reprojection_loss.mean()


class RegularizerLoss(BaseLoss):
    def __init__(self):
        super(RegularizerLoss, self).__init__()

    def forward(self, a, b):
        loss = ((a - 1) ** 2 + b ** 2)
        loss = loss.sum(1, True)
        return torch.mean(loss)


class abLoss(BaseLoss):
    """STFT warping loss using affine transform parameters."""

    def __init__(self):
        super(abLoss, self).__init__()

    def warp_stft(self, audio_2, a, b):
        a_f, a_t = a[0], a[1]
        b_f, b_t = b[0], b[1]
        b, f_bins, time_steps = audio_2.shape

        f = torch.arange(f_bins, device=a_f.device).float()
        t = torch.arange(time_steps, device=a_t.device).float()

        f_prime = (a_f * f + b_f).squeeze()
        t_prime = (a_t * t + b_t).squeeze()
        f_prime = torch.clamp(f_prime, 0, f_bins - 1)
        t_prime = torch.clamp(t_prime, 0, time_steps - 1)

        f_grid, t_grid = torch.meshgrid(f_prime, t_prime, indexing="ij")
        grid = torch.stack((f_grid, t_grid), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        warped_stft = F.grid_sample(
            audio_2.to(a_f.device).unsqueeze(1),
            grid, align_corners=True, mode="bilinear"
        ).squeeze(1)
        return warped_stft

    def forward(self, audio, audio_2, a, b):
        warped_stft_L = self.warp_stft(
            audio[:, 0], (a[:, 0]), (b[:, 0])).unsqueeze(1)
        warped_stft_R = self.warp_stft(
            audio[:, 1], (a[:, 1]), (b[:, 1])).unsqueeze(1)
        warped_stft = torch.cat((warped_stft_L, warped_stft_R), dim=1)
        return torch.mean(torch.pow(audio_2.to(a[0].device) - warped_stft, 2))


class SSIMLoss(nn.Module):
    """Structural similarity loss."""

    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, pred, gt):
        self.ssim.to(pred.device)
        ssim_val = self.ssim(pred, gt)
        return ssim_val.mean()

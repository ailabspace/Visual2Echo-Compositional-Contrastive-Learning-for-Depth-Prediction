import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LaplacianDepthLoss(nn.Module):
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


class ContrastiveLoss(BaseLoss):
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
    def __init__(self, temperature=1.0):
        super(CosineCCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_t, f_s):
        f_t = f_t.reshape(f_t.size(0), -1)
        f_s = f_s.reshape(f_s.size(0), -1)
        f_t_n = F.normalize(f_t, p=2, dim=1)
        f_s_n = F.normalize(f_s, p=2, dim=1)
        sim = (f_t_n * f_s_n).sum(dim=1) / self.temperature
        loss = (1.0 - sim).mean() + (1.0 - sim).mean()
        return loss


class SILogLoss(BaseLoss):
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


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, pred, gt):
        self.ssim.to(pred.device)
        ssim_val = self.ssim(pred, gt)
        return ssim_val.mean()


class FeatureStdHinge(nn.Module):
    def __init__(self, gamma=0.5, eps=1e-4):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, feat):
        if feat.dim() > 2:
            z = feat.flatten(1)
        else:
            z = feat
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return F.relu(self.gamma - std).mean()

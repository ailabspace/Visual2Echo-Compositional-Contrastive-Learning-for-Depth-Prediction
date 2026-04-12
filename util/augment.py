import torch
import torch.nn.functional as F
import random


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.5, std=0.005):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, audio):
        noise = torch.randn_like(audio) * self.std + self.mean
        return audio + noise


class VolumeJitter(torch.nn.Module):
    def __init__(self, jitter_range=(0, 0.05)):
        super(VolumeJitter, self).__init__()
        self.jitter_range = jitter_range

    def forward(self, audio):
        jitter_factor = random.uniform(
            1.0 + self.jitter_range[0], 1.0 + self.jitter_range[1])
        return audio * jitter_factor


class ChannelFlip(torch.nn.Module):
    """Flip left/right audio channels and horizontally flip depth."""

    def __init__(self):
        super(ChannelFlip, self).__init__()

    def forward(self, inputs):
        audio = inputs[0]
        depth_image = inputs[1]

        audio = audio.flip(0)
        depth_image = depth_image.flip(-1)
        if len(inputs) == 3:
            occupancy = inputs[2]
            occupancy = occupancy.flip(-1)
            return audio, depth_image, occupancy
        else:
            return audio, depth_image


class Trimming(torch.nn.Module):
    """Trim audio and reduce depth range proportionally."""

    def __init__(self, min_trim_percent=0.1, max_trim_percent=0.2):
        super(Trimming, self).__init__()
        self.max_trim_percent = max_trim_percent
        self.min_trim_percent = min_trim_percent

    def forward(self, inputs):
        audio = inputs[0]
        depth_image = inputs[1]

        trim_percent = random.uniform(self.min_trim_percent, self.max_trim_percent)
        num_samples = audio.size(1)
        trim_samples = int(num_samples * trim_percent)
        audio[:, :-trim_samples] = 0

        max_depth = depth_image.max().item()
        reduction_amount = max_depth * trim_percent
        trimmed_depth = torch.clamp(depth_image, max=max_depth - reduction_amount)
        if len(inputs) == 3:
            return audio, trimmed_depth, inputs[-1]
        else:
            return audio, trimmed_depth


class Deafening(torch.nn.Module):
    """Zero out one audio channel and apply directional gradient to depth."""

    def __init__(self):
        super(Deafening, self).__init__()

    def deafen(self, audio, channel):
        audio[channel, :] = 0
        return audio

    def gradient_to_img(self, image, side):
        _, height, width = image.size()
        if side == 'left':
            gradient = torch.linspace(0.6, 1., steps=height,
                                      device=image.device).unsqueeze(0).expand(height, width)
        elif side == 'right':
            gradient = torch.linspace(1., 0.6, steps=height,
                                      device=image.device).unsqueeze(0).expand(height, width)
        else:
            raise ValueError("side must be 'left' or 'right'")
        return image * gradient

    def forward(self, inputs):
        audio = inputs[0]
        depth_image = inputs[1]
        channel = random.randint(0, 1)

        audio = self.deafen(audio, channel)
        side = 'left' if channel == 0 else 'right'

        depth_image = self.gradient_to_img(depth_image, side)
        if len(inputs) == 3:
            occupancy_grid = self.gradient_to_img(inputs[2], side)
            return audio, depth_image, occupancy_grid
        return audio, depth_image


class RandomApplyTransform(torch.nn.Module):
    """Randomly apply one of the given transforms with probability p."""

    def __init__(self, transforms, p=0.5):
        super(RandomApplyTransform, self).__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, inputs):
        if random.random() >= self.p:
            return inputs
        transform = random.choice(self.transforms)
        return transform(inputs)

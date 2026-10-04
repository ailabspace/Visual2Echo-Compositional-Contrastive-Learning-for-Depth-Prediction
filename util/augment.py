import torch


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.5, std=0.005):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, audio):
        noise = torch.randn_like(audio) * self.std + self.mean
        return audio + noise

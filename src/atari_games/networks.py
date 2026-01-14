from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQNCNN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.features(x)
        return self.head(x)


class DuelingDQNCNN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int, noisy: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        linear = NoisyLinear if noisy else nn.Linear
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            linear(64 * 7 * 7, 512),
            nn.ReLU(),
            linear(512, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Flatten(),
            linear(64 * 7 * 7, 512),
            nn.ReLU(),
            linear(512, num_actions),
        )
        self.noisy = noisy

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        feats = self.features(x)
        value = self.value_stream(feats)
        adv = self.adv_stream(feats)
        return value + adv - adv.mean(dim=1, keepdim=True)


class DQNMLP(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_sizes: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class DuelingDQNMLP(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_sizes: tuple[int, int] = (128, 128), noisy: bool = False) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        linear = NoisyLinear if noisy else nn.Linear
        self.value_stream = nn.Sequential(
            linear(input_dim, h1),
            nn.ReLU(),
            linear(h1, h2),
            nn.ReLU(),
            linear(h2, 1),
        )
        self.adv_stream = nn.Sequential(
            linear(input_dim, h1),
            nn.ReLU(),
            linear(h1, h2),
            nn.ReLU(),
            linear(h2, num_actions),
        )
        self.noisy = noisy

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        return value + adv - adv.mean(dim=1, keepdim=True)

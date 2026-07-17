# This module performs the forward process of Stable Diffusion
import torch.nn as nn
import torch
import utils

class NoiseScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def linear_beta_schedule(self):
        scale = 1000 / self.timesteps
        start = scale * 0.0001
        end = scale * 0.02
        return torch.linspace(start, end, self.timesteps, dtype=torch.float64)


def _extract(a, t, x_shape):
    # a: 1-D tensor of length T, indexed at t (shape (b,)), then reshaped
    # to (b, 1, 1, 1, ...) so it broadcasts against an (b, c, h, w) image.
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class ForwardProcess(nn.Module):
    def __init__(
        self,
        noise_scheduler=NoiseScheduler(1000),
        steps = 1000, # number of diffusion steps
        visualize=False, # Plot the diffusion process during forward process
    ):
        super().__init__()
        self.steps = steps
        self.visualize = visualize

        betas = noise_scheduler.linear_beta_schedule()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

        # 用 register_buffer 存常数：不是可训练参数，但要跟着 .to(device) 走
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

    def forward(self, x0, t, noise=None):
        # x0: (b, c, h, w) 原图；t: (b,) 每张图各自的时间步
        # 对应笔记里的重参数化闭式解:
        #   x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        if self.visualize:
            utils.show_image(x_t.squeeze())
        return x_t, noise
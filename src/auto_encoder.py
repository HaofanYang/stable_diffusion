# Ideally, we should have a functional image encoder and decoder to reduce the dimension of the input image
# Given we are working with small dataset and it should be fine to not encode at all
# Here, we are creating a "dummy" encoder that just flattens the input image
import torch.nn as nn
import torch

class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return x

class DummyDecoder(nn.Module):
    def __init__(self, shape):
        # shape: 原始图片的 (C, H, W)，用来把 DummyEncoder 展平后的向量 reshape 回去
        super().__init__()
        self.shape = shape

    def forward(self, x):
        b = x.shape[0]
        return x.reshape(b, *self.shape)

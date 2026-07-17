## This file defines useful modules such as Res, Upsample, Downsample etc
import torch.nn as nn

class Residule(nn.module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x + x
        return x
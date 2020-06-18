# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(Loss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()



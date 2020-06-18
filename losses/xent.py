# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> xent.py
@Date : Created on 2020/4/30 14:27
@author: wanghao
========================"""

import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Loss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.ce(input, target)
        return loss

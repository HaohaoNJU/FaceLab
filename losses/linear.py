# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> linear.py
@Date : Created on 2020/5/15 10:20
@author: wanghao
========================"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
import math

def l2_norm(input,axis=1):

    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class MetricFunc(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, in_features, out_features,kernel_norm=False,**kwargs):
        super(MetricFunc, self).__init__()
        self.classnum = out_features
        self.kernel = Parameter(torch.Tensor(in_features,out_features))
        # initial kernel
        init.xavier_uniform_(self.kernel)
        self.kernel_norm = kernel_norm
        # weights norm
        if self.kernel_norm:
            self.kernel = l2_norm(self.kernel,axis=0)
            
    def forward(self, embbedings,label):
        output = torch.mm(embbedings,self.kernel)
        return output




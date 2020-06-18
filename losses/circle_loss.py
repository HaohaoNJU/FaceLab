# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> circle_loss.py
@Date : Created on 2020/5/6 14:38
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
    # implementation of circle loss in 《Circle Loss: A Unified Perspective of Pair Similarity Optimization》
    def __init__(self, in_features, out_features, s=0.5, m=0.5):
        """
        :param in_features: embedding dim
        :param out_features: num of classes
        :param s: defined parameter gamma
        :param m: defined parameter margin
        """
        super(MetricFunc, self).__init__()
        self.classnum = out_features
        self.kernel = Parameter(torch.Tensor(in_features,out_features))
        # initial kernel
        init.xavier_uniform_(self.kernel)
        self.m = m
        self.s = s
    def forward(self, embbedings,label):
        embbedings = l2_norm(embbedings)
        # kernel normalization
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosines = torch.mm(embbedings,kernel_norm)

        label = label.reshape(-1,1).long()
        # calculate alpha_p and alpha_n
        alpha = cosines + self.m
        alpha_p = -torch.gather(cosines,index=label,dim=1) + self.m + 1
        alpha.scatter_(dim=1,index=label,src=alpha_p)

        # calculate margin_p and margin_n
        margin = self.m * torch.ones_like(cosines,device=embbedings.device)
        margin_p = 1-torch.gather(margin,index=label,dim=1)
        margin.scatter_(dim=1,index=label,src=margin_p)
        return (cosines-margin)*alpha*self.s


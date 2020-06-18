# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> multi_logistic.py
@Date : Created on 2020/4/30 11:53
@author: wanghao
========================"""


import torch
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self, gamma=0.5,num_classes=1000,top_k = 0):
        super(Loss, self).__init__()
        self.gamma = gamma   # weight between positives and negatives
        self.sigmoid = torch.nn.Sigmoid()
        self.num_classes = num_classes
        self.top_k = top_k   # how many hardest negatives to be considered, setting 0 will include all
        self.index =torch.arange(num_classes).reshape(1,-1).long() # 1 x C
    def forward(self, input, target):
        batch_size = len(input)
        target=target.reshape(-1,1).long() # B x 1
        sigmoid = self.sigmoid(input) # B x C
        index = self.index.repeat((batch_size,1)).to(input.device) # B x C
        p_mask = (index==target) # B x C
        n_mask = (index!=target) # B x C
        p_sigmoid = sigmoid[p_mask].clamp_min_(0.001)
        n_sigmoid = sigmoid[n_mask].clamp_max_(0.999)
        pos_loss = -torch.log(p_sigmoid).mean()
        if self.top_k:
            n_sigmoid, _ = torch.topk(n_sigmoid.view(batch_size, self.num_classes - 1), k=self.top_k, dim=-1, largest=True)
        neg_loss = -torch.log(1 - n_sigmoid).mean()
        loss = pos_loss * self.gamma + neg_loss * (1 - self.gamma)
        return loss.mean()


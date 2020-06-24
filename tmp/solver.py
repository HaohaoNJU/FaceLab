# -*- coding: utf-8 -*-
"""========================
@Project -> File : projects -> solver.py
@Date : Created on 2020/4/18 11:36
@author: wanghao
========================"""
import torch
from torch import nn
from torch import optim
import datetime
import math
import os
import pdb
import shutil
import gc
import time
from tensorboardX import SummaryWriter
import seaborn as sbn
from matplotlib import pyplot as plt
plt.switch_backend("agg")
import numpy as np
import pandas as pd
from torch.autograd import Variable
import tqdm
import argparse
from utils import get_func, parse_args, separate_fc_params, init_func_and_tag
from modules import IOFactory,OptimFactory,Header
from torch.optim.lr_scheduler import StepLR
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel


class Solver(nn.Module):
    def __init__(self, opt=None):
        super(Solver,self).__init__()
        self.net = get_func(opt.backbone)
        self.backbone_kwargs = opt.backbone_kwargs
        self.net_tag = opt.backbone.split(".")[-1]
        self.embedding_dim = opt.embedding_dim
        self.num_classes = opt.num_classes

        # depends on classifier or pair wise learning
        self.pairwise_learning = opt.pairwise_learning
        self.header_cfg = opt.pairwiser if opt.pairwise_learning else opt.classifier

        self.use_gpu = opt.use_gpu
        self.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")] if isinstance(opt.gpu_ids,str)\
            else [int(x) for x in opt.gpu_ids]

        self.batch_size = opt.batch_size
        self.print_freq = int(opt.print_freq)
        self.val_interval = opt.val_interval

        self.dist_parallel = opt.dist_parallel
        self.optimizer = opt.optimizer
        self.epoch = opt.epoch
        self.start_epoch = opt.start_epoch
        # lr and other hyper params
        self.rigid_lr = opt.rigid_lr
        self.lr_decay = opt.lr_decay
        self.hypers = opt.hypers

        self.dataset_loader = get_func(opt.dataset_loader)
        self.train_data_kwargs = opt.train_data_kwargs
        self.validate = opt.validate

        self.train_data = opt.train_data
        self.val_data = opt.val_data

        self.visualize = opt.visualize
        self.vis_kwargs = opt.vis_kwargs
        self.pretrained = opt.pretrained
        self.save_path = opt.save_path
        self.model_weight_path = opt.model_weight_path
        self.date_tag = str(datetime.datetime.today())[:10]  # to mark save files
        #########################################################################
        print("MODEL BUILDING UP . . .")
        self.build_model()

    def init_model(self):
        # backbone init
        self.net = self.net(**self.backbone_kwargs)
        # header init
        self.header = Header(
            pairwise=self.pairwise_learning,
            net_tag=self.net_tag,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            config=self.header_cfg
        )
        # optimizer init
        self.optim_fac = OptimFactory(
            params = self.net,
            rigid_lr=self.rigid_lr,
            mix_prec = self.dist_parallel,
            **self.hypers
        )
        # io factory init
        self.io_fac = IOFactory(
            save_path = self.save_path,
            tag = self.header.tag,
            vis=self.visualize,
            log_name="train"
        )
        if self.dist_parallel:
            distributed.init_process_group(
                "nccl", init_method="env://"
            )

    def build_model(self):
        # data loader init
        print("Data Loader building up . . .")
        self.train_loader = self.dataset_loader(data_path=self.train_data,
                                                batch_size=self.batch_size,
                                                dist_parallel = self.dist_parallel,
                                                **self.train_data_kwargs)

        self.val_loader = self.dataset_loader(self.val_data) if self.validate else None
        self.num_classes = self.train_loader.num_classes
        print("MODULE INITIALIZATION . . .")
        self.init_model()
        # load weight
        if self.pretrained:
            if self.model_weight_path.startswith("http"): # load from url
                self.net.load_state_dict(torch.hub.load_state_dict_from_url(self.model_weight_path))
            else:
                state_dict = torch.load(self.model_weight_path)
                self.net.load_state_dict(state_dict)
            self.header.load()
        # add params
        for module,_,_,optim_name,hypers in self.header.learnable_modules:
            self.optim_fac.add_optim(
                params = module,
                optim_name = optim_name,
                **hypers
            )

        # cuda and parallel
        if not torch.cuda.is_available() or not self.use_gpu:
            self.net = self.net.cpu()
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:{}".format(self.gpu_ids[0]))
            self.net = self.net.to(self.device)
            self.header.to(self.device)
            if len(self.gpu_ids) > 1 and self.dist_parallel:
                self.net = convert_syncbn_model(self.net)
                self.net = DistributedDataParallel(self.net,delay_allreduce=True)
                self.header.dist_parallel()
            elif len(self.gpu_ids) > 1:
                self.net = nn.DataParallel(self.net,device_ids=self.gpu_ids)
                self.header.parallel(self.gpu_ids)

    def heatmapper(self,embeddings=None,labels=None,steps=None):
        with torch.no_grad():
            fig = plt.figure(figsize=self.vis_kwargs["hm_size"])
            embs = embeddings.data.cpu().numpy()
            labs = labels.data.cpu().numpy()
            mat = np.dot(embs, embs.T)
            data_frame = pd.DataFrame(np.around(mat, decimals=4), columns=labs,index=labs)
            sbn.heatmap(data_frame, annot=False)
            self.io_fac.writer.add_figure("heatmap", figure=fig, global_step=(steps))

    def train(self):
        self.io_fac.logging("Model Loaded from {} ".format(self.model_weight_path))
        steps = self.start_epoch*self.train_loader.__len__
        self.io_fac.logging("START TRAINING!!!")
        for epoch in range(self.start_epoch,self.start_epoch+self.epoch):
            self.net.train()
            self.optim_fac.lr_step(epoch,self.lr_decay)
            # scheduler.step()
            tic = time.time()
            acc_sum = 0 ; loss_sum = 0

            epoch_steps = self.train_loader.__len__()
            if epoch != self.start_epoch:
                print("Data Loader Reloading . . .")
                self.train_loader.reload()
            for optim in self.optim_fac.optimizers:
                state = optim.state_dict()["param_groups"][0]
                state = ["{} : {}\n".format(k, v) for k, v in state.items() if k !="params"]
                self.io_fac.logging(state)

            for i,(imgs,labels) in enumerate(self.train_loader.sample_loader):

                imgs = imgs.to(self.device)
                labels = labels.to(self.device).long()
                embeddings = self.net(imgs)
                embeddings.squeeze_()
                (loss,outputs) = self.header(embeddings,labels)
                loss = loss.mean()
                self.optim_fac.reset()
                if self.dist_parallel:
                    self.optim_fac.dist_step(loss)
                else:
                    loss.backward()
                    self.optim_fac.step()

                if self.visualize:
                    self.io_fac.writer.add_scalar("scalar/loss", loss.item(), steps)

                steps += 1
                if steps % self.print_freq == 0:
                    toc = time.time()
                    loss_val = loss.data.item()
                    # output = torch.argmax(output.data)
                    # acc = (output==label).float().mean().data.item()*100 # scalor
                    # acc_sum += acc
                    loss_sum += loss_val
                    if self.visualize:
                        self.io_fac.writer.add_scalar("scalar/net_lr", self.optim_fac.optimizers[0].param_groups[0]["lr"],steps)
                        if steps % (self.print_freq * self.vis_kwargs["emb_delay"]) ==0 :
                            self.heatmapper(embeddings,labels,steps)
                            self.io_fac.writer.add_embedding(
                                mat=embeddings,
                                metadata=labels,
                                label_img=imgs,
                                global_step=steps
                            )
                            self.io_fac.flush()
                    self.io_fac.logging(
                                 "epoch {} iter {}/{}, loss = {:.4}, time_span = {}".\
                                 format(epoch,i,epoch_steps,loss_val,np.around(toc-tic,1)))
                    tic = time.time()
            time_str = time.asctime(time.localtime(time.time()))
            acc_mean = acc_sum/((epoch_steps-1)//self.print_freq+1)
            loss_mean = loss_sum/((epoch_steps-1)//self.print_freq+1)
            self.io_fac.logging("{}".format(time_str),
                         "********EPOCH {}, LOSS_MEAN = {:.4}, ACC_MEAN = {:.3}%********".format(epoch, loss_mean, acc_mean))
            if epoch % self.val_interval == 0 or epoch == self.start_epoch + self.epoch - 1:
                save_dir = os.path.join(self.io_fac.save_weight_path,"epoch{}_steps{}.pth".format(epoch,steps))
                torch.save(self.net.module.state_dict(),save_dir)
                self.io_fac.logging("Model Saved into {}".format(save_dir))
                self.header.save(epoch,self.io_fac.save_weight_path)
                if self.validate:
                    self.val()
    def val(self):
        pass # Online validation is to be added !

if __name__ == "__main__":
    # get config
    from easydict import EasyDict
    import yaml
    args = parse_args()
    # opt = args.cfg[:-3] if args.cfg.endswith("py") else args.cfg
    # opt = get_func(opt)
    opt = EasyDict(yaml.load(open(args.cfg,"r"),Loader = yaml.Loader))
    # specify gpus to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # initialize and run up !
    solver = Solver(opt)
    solver.train()


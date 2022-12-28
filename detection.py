from models.yolo import  Model
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from datasets import *
import numpy as np

import os.path as osp
import time
import sys
import logging
from test import test
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from torch import optim
from optimizer import Optimizer
import yaml
import os.path as osp
import os
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from general import *
from loss import *
from utils.loss import *
import datetime
import  torch

def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

def train_dete(i,logger=None):
    device = torch.device('cuda')
    hyp = 'data/hyp.scratch.yaml'
    opt = 'data/voc.yaml'
    load_path = ''
    train_path=''
    modelpth = './model'
    Method = 'Fusion'
    save_dir='runs/train'
    modelpth = os.path.join(modelpth, Method)
    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    with open(opt) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    names = data_dict['names']  # class names
    train_path = data_dict['train']
    test_path = data_dict['val']
    model = Model(ch=3, nc=5, anchors=hyp.get('anchors'))

    ema = ModelEMA(model)



    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in [640,640]]  # verify imgsz are gs-multiples

    dataloader, dataset = create_dataloader(train_path, 640, 16, gs,
                                            hyp=hyp, augment=True, prefix=colorstr('train: '),split='none')

    testloader = create_dataloader(train_path, 640, 8 * 2, gs,   # testloader
                                   hyp=hyp, rect=True, rank=-1,
                                   pad=0.5, prefix=colorstr('val: '),split='none')[0]

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches

    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= 5 / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = float(0.0)
    model.nc = 5  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(dataset.labels, 5).to(device) * 5
    model.class_weights = labels_to_class_weights(dataset.labels, 5) * 5# attach class weights
    model.names = names
    if i >0:
        model.load_state_dict(torch.load(load_path))

    # model.cuda()
    model.train()
    compute_loss = ComputeLoss(model)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 800
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i * 200
    iter_nums = 300

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optim = torch.optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    optim.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optim.add_param_group({'params': pg2})  # add pg2 (biases)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2


    #train-loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter  = enumerate(dataloader)
    epoch = 0
    for it in range(iter_nums):
        for i, (imgs, targets, paths, _) in diter:
            # imgs = imgs.to(device, non_blocking=True).float() / 255.0
            imgs = imgs.float() / 255.0
            print(imgs.shape)
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
            loss.backward()
            optim.step()

            loss_avg.append(loss.item())
            if (it + 1) % msg_iter == 0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                lr = optim.lr
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                eta = int((max_iter - it) * (glob_t_intv / i))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(
                    [
                        'it: {it}/{max_it}',
                        'lr: {lr:4f}',
                        'loss: {loss:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=it_start + i + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
                )
                logger.info(msg)
                loss_avg = []
                st = ed

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        final_epoch = it + 1 == iter_nums
        results, maps, times = test(data_dict,
                                             batch_size=1,
                                             imgsz=imgsz_test,
                                             model=ema.ema,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             verbose=5 < 50 and final_epoch,
                                             plots=False,

                                             compute_loss=compute_loss,
                                             is_coco=None)

    save_pth = osp.join(modelpth, 'model_final.pth')
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, save_pth)
    # logger.info(
    #     'Segmentation Model Training done~, The Model is saved to: {}'.format(
    #         save_pth)
    # )
    # logger.info('\n')


if __name__ == "__main__":
    # logpath = '/logs'
    # logger = logging.getLogger()
    # setup_logger(logpath)

    train_dete(i=0,)
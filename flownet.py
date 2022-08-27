import numpy as np
import torch
import sys
import torch.nn as nn
import pathlib
import sys
import os
import argparse
import pdb
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image

resample_method = 'border'

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

def grid_sample(input1, input2, mode='bilinear', align_corners=True):    
    return torch.nn.functional.grid_sample(input1, input2, mode=mode, padding_mode=resample_method, align_corners=align_corners)

def resample(image, flow, mode='bilinear'):        
    b, c, h, w = image.size()        
    grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
    #print(flow.size())
    final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
    #print("final_grid", final_grid.size())
    output = grid_sample(image, final_grid, mode, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(image.size())).cuda()
    mask = grid_sample(mask, final_grid, align_corners=True)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask


# normalize flow + warp = resample
def warp(im, flow, padding_mode='border'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''
    warped = F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped

class RAFT(nn.Module):
    def __init__(self, model='kitti'):
        super(RAFT, self).__init__()
        sys.path.append("../")
        from raft import raft

        if model == 'things':
            model = 'raft-things.pth'
        elif model == 'kitti':
            model = 'raft-kitti.pth'
        elif model == 'chairs':
            model = 'raft-chairs.pth'
        elif model == 'sintel':
            model = 'raft-sintel.pth'


        # TODO: Figure out how to do checkpoints
        raft_dir = pathlib.Path('../raft/models/')

        # Emulate arguments
        args = argparse.Namespace()
        args.model = raft_dir / model
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False
        #args.alternate_corr = True # TODO: This doesn't work :(

        flowNet = nn.DataParallel(raft.RAFT(args))
        flowNet.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("loading model", args.model)
        self.flowNet = flowNet.module
        for param in self.flowNet.parameters():
            param.requires_grad = False 

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''

        # Normalize to [0, 255]
        #with torch.no_grad():
        # rescale [-1,1] to [0,1]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        flow_low, flow_up = self.flowNet(im1, im2, iters=10, test_mode=True)

        im1 = im1 / 255.0 
        im2 = im2 / 255.0
        
        conf = (self.norm(im1 - resample(im2, flow_up)) < 0.01).float()
        #torch.ones_like(flow_up)#
        return flow_up, conf 
    
    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)  
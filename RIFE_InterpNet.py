import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from models.RIFE.model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.RIFE.train_log.IFNet_HDv3 import *
import torch.nn.functional as F
#from model.loss import *
# parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--img', dest='img', nargs=2, required=True)
# parser.add_argument('--exp', default=1, type=int)
# parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
# parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
# parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
# parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

# args = parser.parse_args()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# class RIFE_HDv3:
#     def __init__(self, local_rank=-1):
#         self.flownet = 
#         self.device()

    
        
#     def save_model(self, path, rank=0):
#         if rank == 0:
#             torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

#     def inference(self, img0, img1, scale=1.0):
#         imgs = torch.cat((img0, img1), 1)
#         scale_list = [4/scale, 2/scale, 1/scale]
#         flow, mask, merged = self.flownet(imgs, scale_list)
#         return merged[2]

class RIFE_InterpNet(nn.Module):
    def __init__(self):
        super(RIFE_InterpNet, self).__init__()
        self.model = IFNet()
        self.load_model("./models/RIFE/train_log/", -1)
        print("Loaded v3.x HD model.")
        for param in self.parameters():
            param.requires_grad = False

    
    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.model.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            else:
                self.model.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')))
        
    def forward(self, im0, im1):
        im0 = (im0 + 1.0)/2.0
        im1 = (im1 + 1.0)/2.0
        scale = 1.0
        # n, c, h, w = img0.shape
        # ph = ((h - 1) // 32 + 1) * 32
        # pw = ((w - 1) // 32 + 1) * 32
        # padding = (0, pw - w, 0, ph - h)
        # img0 = F.pad(img0, padding)
        # img1 = F.pad(img1, padding)
        #mid = model.inference(im0, im1)
        imgs = torch.cat((im0, im1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.model(imgs, scale_list)
        
        #for i in range(3):
        #    print(flow[i].size())
        mid = merged[2]
        mid = mid * 2.0 - 1.0
        mid = torch.clamp(mid, -1.0, 1.0)
        return mid, flow[-1][:,:2,...]

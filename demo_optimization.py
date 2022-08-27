from ast import parse
import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import warnings
from skimage.transform import resize as imresize
from skimage.io import imread
from torch.autograd import Variable
from math import ceil
import glob
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
from model.warplayer import warp
from torchvision.utils import save_image
from torch.nn import Parameter
#from torch.nn.parallel import DistributedDataParallel as DDP
from train_log.IFNet_HDv3 import *
import pytorch_ssim
#from model.loss import *
import matplotlib.pyplot as plt
from optimization_utils import fwd2bwd
from torch.nn.functional import sigmoid
import time 
from region_fill import regionfill
from PIL import Image 

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', nargs=2, required=False)
parser.add_argument('--exp', default=1, type=int)
parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

# for configurations
parser.add_argument('--val_dir', type=str, default='the directory of validation dataset')
parser.add_argument('--index', type=int, default=0, help='the index number of the first image')
parser.add_argument('--start_id', type=int, default=0, help='the start video id')
parser.add_argument('--length', type=int, default=10, help='the start video id')
parser.add_argument('--input_num', type=int, default=2, help='the length of the input frames')
parser.add_argument('--pred_num', type=int, default=4, help='the number of predicted frames')
parser.add_argument('--output_dir', type=str, default='', help='The path of the gt image')

parser.add_argument('--consist_weight', default=0.1, type=float)
parser.add_argument('--lr_stage1', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_stage2', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_flow_consistency', action='store_true', help='whether use flow consistency loss')
parser.add_argument('--use_lr_decay', action='store_true', help='whether use learning rate decay policy')
parser.add_argument('--use_smoothness', action='store_true', help='whether use smoothness loss')
parser.add_argument('--use_interp_im2', action='store_true', help='whether compare the interp result with im2')
parser.add_argument('--use_warp_im2', action='store_true', help='whether compare the warped pred with im3')
parser.add_argument('--optimize_frame', action='store_true', help='in the later stage, optimize a frame')
parser.add_argument('--tv', action='store_true', help='The total variantional loss')
parser.add_argument('--stage1_iters', default=2000, type=int, help='the iterations in first stage')
parser.add_argument('--stage2_iters', default=2000, type=int, help='the iterations in second stage. Only useful when optimize_frame=True')
parser.add_argument('--dataset', type=str, default='cityscapes', help='choose from datasets: cityscapes/kitti/vimeo/middlebury/ucf')
parser.add_argument('--height', default=256, type=int)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--scale', default=16, type=int, help='rescale image')
parser.add_argument('--initial', default='warp', type=str, help='choose initilization of optical flow, warp or copy')
parser.add_argument('--flow_ckpt', default='kitti', type=str, help='choose the ckpt type of flow model, kitti or sintel')
parser.add_argument('--pred_steps', default=5, type=int, help='the number of predicted steps')
parser.add_argument('--first_frame', default=2, type=int, help='the index of first frame')
parser.add_argument('--second_frame', default=3, type=int, help='the index of second frame')
parser.add_argument('--warp_weight', default=1.0, type=float, help='weight for warp im')
parser.add_argument('--interp_weight', default=1.0, type=float, help='interp weight')
args = parser.parse_args()
output_dir = args.output_dir

from flownet import RAFT


dataset = args.dataset
input_num = args.input_num
pred_num = args.pred_num
val_dir = args.val_dir
index = args.index

if args.flow_ckpt == 'kitti':
    flow_method = RAFT(model='kitti')
else:
    flow_method = RAFT(model='sintel')


output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=True)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def t2n(tensor):
    # tensor2numpy
    return tensor.detach().cpu().numpy()

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def gradient_x(self, img):
        gx = (img[:,:,:-1,:] - img[:,:,1:,:])
        return gx

    def gradient_y(self, img):
        gy = (img[:,:,:,:-1] - img[:,:,:,1:])
        return gy

    def compute_smooth_loss(self, flow_x, img):
        flow_gradients_x = self.gradient_x(flow_x)
        flow_gradients_y = self.gradient_y(flow_x)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, True))

        smoothness_x = flow_gradients_x * weights_x
        smoothness_y = flow_gradients_y * weights_y

        return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(flow[:,i:i+1,:,:], img)
        return smoothness/2

    def forward(self, flow, image):
        return self.compute_flow_smooth_loss(flow, image)


class Consistency(nn.Module):
    # Consistency loss for a pair of optical flow
    def __init__(self):
        super(Consistency, self).__init__()
        self.beta = 0.05
        self.weight = 0.001

    def L2_norm(self, x): 
        return F.normalize(x, p=2, dim=1, eps=1e-12)#.unsqueeze(1)

    def forward(self, flow_fwd, flow_bwd, stage_num):
        devide = flow_fwd.get_device()
        alpha = torch.FloatTensor([1.5]).cuda(devide)

        bwd2fwd_flow_pyramid = resample(flow_bwd, flow_fwd)# From bwd coordinate to src coordinate
        fwd2bwd_flow_pyramid = resample(flow_fwd, flow_bwd)# From fwd coordinate to tgt coordinate
        #print("bwd2fwd_flow_pyramid", bwd2fwd_flow_pyramid.size())
        fwd_diff = torch.abs(bwd2fwd_flow_pyramid + flow_fwd)# In src
        bwd_diff = torch.abs(fwd2bwd_flow_pyramid + flow_bwd)# In tgt
        #print("fwd_diff size = ", fwd_diff.size())
        fwd_consist_bound = self.beta * self.L2_norm(flow_fwd) 
        bwd_consist_bound = self.beta * self.L2_norm(flow_bwd) 
        #print("fwd_consist_bound = ", fwd_consist_bound.size())
        fwd_consist_bound = alpha.clone().detach()#torch.max(fwd_consist_bound, alpha).clone().detach()
        #bwd_consist_bound = torch.max(bwd_consist_bound, alpha).clone().detach()
        bwd_consist_bound = alpha.clone().detach()
        fwd_mask = (fwd_diff < fwd_consist_bound).float()# In src
        bwd_mask = (bwd_diff < bwd_consist_bound).float()# In tgt

        if stage_num == 2:
            flow_consistency_loss = self.weight/2 * \
                (torch.sum(torch.mean(fwd_diff, dim=1, keepdim=True) * fwd_mask)  + \
                torch.sum(torch.mean(bwd_diff, dim=1, keepdim=True) * bwd_mask))
            #(torch.sum(torch.mean(fwd_diff, dim=1, keepdim=True))
        else:
            flow_consistency_loss = self.weight/2 *  (\
             torch.sum(torch.mean(bwd_diff, dim=1, keepdim=True)))
        return fwd_mask, bwd_mask, flow_consistency_loss



class RIFE_InterpNet(nn.Module):
    def __init__(self):
        super(RIFE_InterpNet, self).__init__()
        self.model = IFNet()
        self.load_model("./train_log/", -1)
        print("Loaded interpolation v3.x HD model.")
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
        scale = 1.0
        imgs = torch.cat((im0, im1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        flow, mask, merged, merged_final = self.model(imgs, scale_list)
        mid = merged_final[2]
        mid = torch.clamp(mid, 0.0, 1.0)

        warp_pred = merged[2][1]
        warp_pred = torch.clamp(warp_pred, 0.0, 1.0)
        return mid, flow[-1][:,2:,...], warp_pred

interp_net = RIFE_InterpNet().cuda()


flow_method = flow_method.cuda()

from flownet import resample


# Dataloader for cityscapes
if args.dataset == 'cityscapes':
    cityscapes_root = "../dataset/Cityscapes/images_512x1024/val/"
    city_dir = os.listdir(cityscapes_root)
    city_dir.sort()
    video = []
    for i in range(len(city_dir)):
        frame_dir = cityscapes_root + city_dir[i]
        frame_list = os.listdir(frame_dir)
        frame_list.sort()
        for j in range(len(frame_list)//30):
            image = []
            st = j*30
            dst = (j+1)*30
            for k in range(st, dst):
                full_image_path = frame_dir + "/" + frame_list[k]
                assert os.path.isfile(full_image_path)
                image.append(full_image_path)
            video.append(image)
    num_of_video = len(video)
    print("number of videos is ", num_of_video)
    assert len(video) == 500
elif args.dataset == 'davis':
    # Loading DAVIS dataset
    davis_root = "../dataset/DAVIS/JPEGImages/480p/"
    val_list = '../dataset/DAVIS/ImageSets/2017/val.txt'
    f = open(val_list, 'r')
    files = f.readlines()
    f.close()
    print("len", len(files))
    video = []
    for i in range(len(files)):
        files[i] = files[i][:-1]#remove \n
        image_dir = davis_root + files[i] + "/"
        print(image_dir)
        image_list = glob.glob(image_dir + "*.jpg")
        image_list.sort()
        video.append(image_list)
    num_of_videos = len(video)
    print("number of videos is ", num_of_videos)
    assert num_of_videos == 30
elif args.dataset == 'vimeo':
    # Dataloader for Vimeo
    Vimeo_root = "../dataset/Vimeo/triplet/vimeo_triplet/sequences/"
    testlist = "../dataset/Vimeo/triplet/vimeo_triplet/tri_testlist.txt"
    # readin files
    testlist_f = open(testlist, 'r')
    data = testlist_f.readlines()
    testlist_f.close()
    video = []

    for i in range(len(data)-1):
        seq_id, frame_id = data[i].split("/")
        frame_id = frame_id[:-1]#remove \n
        image_dir = Vimeo_root + seq_id + "/" + frame_id + "/"
        image_list = sorted(glob.glob(image_dir + "*.png"))
        #video.append((image_list, seq_id, frame_id))
        video.append(image_list)
    num_of_videos = len(video)
    print("number of videos is ", num_of_videos)
elif args.dataset == 'middlebury':
    middleburry_root = "../dataset/middlebury/other-data/"

    video = []
    scene_dir = os.listdir(middleburry_root)
    scene_dir.sort()
    for i in range(len(scene_dir)):
        image_dir = middleburry_root + scene_dir[i] + "/"
        image_list = sorted(glob.glob(image_dir + "*.png"))
    video = []
    scene_dir = os.listdir(middleburry_root)
    scene_dir.sort()
    for i in range(len(scene_dir)):
        image_dir = middleburry_root + scene_dir[i] + "/"
        image_list = sorted(glob.glob(image_dir + "*.png"))
        video.append(image_list)
    num_of_videos = len(video)
    print("number of videos is ", num_of_videos)
elif args.dataset == 'kitti':
    kitti_root = "../dataset/KITTI/test_set/"
    video = []
    scene_dir = os.listdir(kitti_root)
    scene_dir.sort()
    for i in range(len(scene_dir)):
        image_dir = kitti_root + scene_dir[i] + "/image_02/data/"
        image_list = os.listdir(image_dir)
        image_list.sort()
        for k in range(len(image_list)-8):
            images = []
            for f in range(k, k + 4):
                image_full_path = image_dir + image_list[f]
                assert os.path.isfile(image_full_path)
                images.append(image_full_path)
            video.append(images)
    num_of_videos = len(video)
    print("number of videos is ", num_of_videos)
    assert num_of_videos == 1337


for video_id in range(args.start_id, args.start_id + args.length):
    cnt_output_dir = output_dir + "%04d/"%video_id
    os.makedirs(cnt_output_dir, exist_ok=True)
    cnt_video_path = video[video_id]

    first_path = cnt_video_path[args.first_frame]
    second_path = cnt_video_path[args.second_frame]

    img = Image.open(first_path)
    w, h = img.size
    if args.width == 0:
        width = w // args.scale * args.scale
        height = h // args.scale * args.scale
    else:
        width = args.width
        height = args.height
    frame1 = TF.to_tensor(imresize(imread(first_path),(height,width))).unsqueeze(0).cuda().float()
    frame2 = TF.to_tensor(imresize(imread(second_path),(height,width))).unsqueeze(0).cuda().float()

    b, _, h, w = frame1.size()


    # gt_frame_list = []
    # for i in range(pred_num):
    #     gt_frame_list.append(TF.to_tensor(imresize(imread(gt_paths[i]),(height,width))).unsqueeze(0).cuda().float())
    #     save_image(gt_frame_list[i] , "./%s/gt_%02d.png"%(output_dir, input_num + i + 1))
    old_lr_stage1 = args.lr_stage1

    
    H = frame1.shape[2]
    W = frame1.shape[3]

    # Loss functions
    l1_loss = nn.L1Loss()
    flow_consistency_loss = Consistency()
    smoothness_loss = SmoothLoss()
    tv_loss = TVLoss()
    index_list = []
    loss_list = []
    ssim_list = []

    save_image(frame1 , "./%s/im1.png"%(cnt_output_dir))
    save_image(frame2 , "./%s/im2.png"%(cnt_output_dir))

    start_time = time.time()
    im_seq_list = []
    im_seq_list.append(frame1)
    im_seq_list.append(frame2)

    for pred_id in range(args.pred_steps):
        # initiliza the vars
        im1 = im_seq_list[-2]
        im2 = im_seq_list[-1]    
        flow_2to1, conf = flow_method(im2, im1)
        flow_2to3 = - flow_2to1.detach().cpu().numpy()

        if args.initial == 'warp':
            bwd_flow_3to2 = fwd2bwd(flow_2to3, conf)

            bwd_flow_3to2 = torch.from_numpy(bwd_flow_3to2).cuda().float()
        elif args.initial == 'copy':
            bwd_flow_3to2 = flow_2to1.clone().float()
        else:
            print("Need to choose a initilization method")

        flow = Variable(bwd_flow_3to2, requires_grad=True)
        optimizer = torch.optim.Adam(
            [{'params': flow, 'lr': old_lr_stage1}]
        )

        for it in range(1, args.stage1_iters + 1):
            optimizer.zero_grad()
            frame_pred = resample(im2, flow)
            interp_result, flow_2to3_pred, warp_pred = interp_net(im1, frame_pred)

            # Loss functions
            loss = 0.0
            if args.use_interp_im2:
                loss_interp = l1_loss(interp_result, im2) * args.interp_weight
                loss += loss_interp
            if args.use_warp_im2:
                loss_warp = l1_loss(warp_pred, im2) * args.warp_weight
                loss += loss_warp
            if args.use_flow_consistency:
                stage_num=1 #if it < 3000 else 2
                fwd_mask, bwd_mask, consist_loss = flow_consistency_loss(flow_2to3_pred, flow, stage_num)
                consist_loss = consist_loss * args.consist_weight
                loss += consist_loss
            if args.use_smoothness:
                smooth_loss = smoothness_loss(flow_2to3_pred, im2)
                smooth_loss = smooth_loss * 5.0
                loss += smooth_loss
            if args.tv:
                t_loss = tv_loss(flow) * 2.0
                loss += t_loss


            loss.backward()
            optimizer.step()


            if it % 50 == 0:
                print_string = "Iter %03d Loss All: %.4f |"%(it, t2n(loss))
                if args.use_interp_im2:
                    print_string += " Interp Loss %.4f |" % t2n(loss_interp)
                if args.use_warp_im2:
                    print_string += " Warp Loss %.4f | " % t2n(loss_warp)
                if args.use_flow_consistency:
                    print_string += " Consist Loss %.4f | " % t2n(consist_loss)
                if args.use_smoothness:
                    print_string += " Smooth Loss %.4f | " % t2n(smooth_loss)
                if args.tv:
                    print_string += " tv Loss %.4f | " % t2n(t_loss)
                print(print_string)
                index_list.append(it)
                loss_list.append(t2n(loss))
                #ssim_list.append(pytorch_ssim.ssim(frame3, gt).detach().cpu().numpy())

            
            if it  == 1:
                initial = frame_pred.detach()
            
            #print(bwd_mask.size())
            if it == args.stage1_iters:
                # apply flow filling here
                frame_pred_before = resample(im2.detach().float(), flow.float())

                im2_vis = im2.detach().cpu().numpy()
                flow_ = flow.detach().cpu().numpy()
                
                bwd_mask_ = 1 - bwd_mask.detach().cpu().numpy().astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                bwd_mask_x = cv2.dilate(bwd_mask_[0,0,...], kernel, iterations=1)
                bwd_mask_y = cv2.dilate(bwd_mask_[0,1,...], kernel, iterations=1)
                flow_x = regionfill(flow_[0,0,...], bwd_mask_x.astype(int))
                flow_y = regionfill(flow_[0,1,...], bwd_mask_y.astype(int))
                flow_new = np.concatenate([flow_x[None,None,...], flow_y[None,None,...]], axis=1)
                flow_new = torch.from_numpy(flow_new).cuda()

                frame_pred = resample(im2.detach().float(), flow_new.float())

                bwd_mask_max, _ = torch.max(bwd_mask, dim=1, keepdim=True)
                bwd_mask_max = bwd_mask_max.detach().cpu().numpy().astype(np.uint8)
                bwd_mask_max = cv2.erode(bwd_mask_max[0,0,...], kernel, iterations=1)
                bwd_mask_max = torch.from_numpy(bwd_mask_max[None, None, ...]).cuda().repeat(1,3,1,1)
                interpolated_region = frame_pred * bwd_mask_max + torch.zeros_like(bwd_mask_max) * (1 - bwd_mask_max)

                #save_image(interp_result , "./%s/Time_%02d_interp_%04d.png"%(output_dir, pred_id, it))
                #save_image(interpolated_region , "./%s/Time_%02d_filling_region_%04d.png"%(output_dir, pred_id, it))
                #save_image(warp_pred , "./%s/Time_%02d_warp_pred_%04d.png"%(output_dir,pred_id,  it))
                save_image(frame_pred.detach() , "./%s/pred_%04d.png"%(cnt_output_dir, pred_id))

                #save_image(frame_pred_before.detach() , "./%s/Time_%02d_pred_before_%04d.png"%(output_dir, pred_id,it))
                #diff = torch.abs(frame_pred.detach()-initial) * 20
                #save_image(diff , "./%s/Time_%02d_diffpred_%04d.png"%(output_dir,pred_id,  it))
                #save_image(fwd_mask, "./%s/Time_%02d_fwdflow_diff_%04d.png"%(output_dir,pred_id,  it))
               # save_image(bwd_mask, "./%s/Time_%02d_bwdflow_diff_%04d.png"%(output_dir,pred_id,  it))
                #print("result ssim", pytorch_ssim.ssim(frame3, gt).detach().cpu().numpy())
                # plt.figure()
                # plt.plot(index_list, loss_list)
                # plt.axis([0, len(index_list), np.min(loss_list), np.max(loss_list)])
                # plt.savefig('./%s/curve_loss.png'%(output_dir))
                # plt.close()
                # plt.figure()
                # plt.plot(index_list, ssim_list)
                # plt.axis([0, len(index_list), np.min(ssim_list), np.max(ssim_list)])
                # plt.savefig('./%s/curve_ssim.png'%(output_dir))
                # plt.close()

                # save flow
                #np.save("./%s/Time_%02d_flow.npy"%(output_dir, pred_id), flow.detach().cpu().numpy())

            if it == args.stage1_iters:
                # last step
                # load newest predicted frame
                im_seq_list.append(frame_pred.detach())


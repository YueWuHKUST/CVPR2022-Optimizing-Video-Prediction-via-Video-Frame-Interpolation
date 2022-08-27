import os
from copy import deepcopy
import torch.nn as nn
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image
from skimage import color
from skimage.measure import compare_psnr, compare_ssim
from torch.autograd import Variable
from skimage.io import imread
from skimage.transform import resize as imresize
import model as models
import dgp_utils as utils
from flownet import resample
from optimization_utils import fwd2bwd
from train_log.IFNet_HDv3 import *
from model.nethook import subsequence

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
        # n, c, h, w = img0.shape
        # ph = ((h - 1) // 32 + 1) * 32
        # pw = ((w - 1) // 32 + 1) * 32
        # padding = (0, pw - w, 0, ph - h)
        # img0 = F.pad(img0, padding)
        # img1 = F.pad(img1, padding)
        #mid = model.inference(im0, im1)
        imgs = torch.cat((im0, im1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        flow, mask, merged, merged_final = self.model(imgs, scale_list)
        mid = merged_final[2]
        mid = torch.clamp(mid, 0.0, 1.0)

        warp_pred = merged[2][1]
        warp_pred = torch.clamp(warp_pred, 0.0, 1.0)
        return mid, flow[-1][:,:2,...], warp_pred




# model to optmize vector directly
class optimize_vector(object):
    def __init__(self, config, flow_method):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.mode = config['dgp_mode']
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']
        self.iterations = config['iterations']
        self.ftr_num = config['ftr_num']
        self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.select_num = config['select_num']
        self.output_dir = config['output_dir']
        # create model
        # Training network of optimization
        height = config['height']
        width = config['width']
        self.frame1 = TF.to_tensor(imresize(imread(config['first']), (height,width))).unsqueeze(0).cuda().float()
        self.frame2 = TF.to_tensor(imresize(imread(config['second']), (height,width))).unsqueeze(0).cuda().float()
        self.img_name = config['first'].split("/")[-1][:-4]
        b, _, h, w = self.frame1.size()

        flow_gt, flow_conf = flow_method(self.frame2, self.frame1)

        if os.path.exists("./%s/flow.npy"%(self.output_dir)):
            bwd_flow_3to2 = np.load("./%s/flow.npy"%(self.output_dir))
            print("loaded previous flow")
        else:
            flow_2to1, conf = flow_method(self.frame2, self.frame1)
            flow_2to3 = - flow_2to1.detach().cpu().numpy()

            bwd_flow_3to2 = fwd2bwd(flow_2to3, conf)

        bwd_flow_3to2 = torch.from_numpy(bwd_flow_3to2).cuda().float()

        self.gt = TF.to_tensor(imresize(imread(config['gt']),(height,width))).unsqueeze(0).cuda().float()

        self.flow = Variable(bwd_flow_3to2, requires_grad=True)
        self.interp_net = RIFE_InterpNet().cuda()
        self.flow_optim = torch.optim.Adam(
            [{'params': self.flow}],
            lr=config['G_lr'],
            betas=(config['G_B1'], config['G_B2']),
            weight_decay=0,
            eps=1e-8)

        # prepare learning rate scheduler
        self.flow_scheduler = utils.LRScheduler(self.flow_optim, config['warm_up'])

        # loss functions
        self.mse = torch.nn.MSELoss()
        vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
        self.ftr_net = subsequence(vgg.features, last_layer='20')
        self.criterion = utils.PerceptLoss()

    def run(self, save_interval=None):
        save_imgs = self.gt.clone()
        save_imgs2 = save_imgs.cpu().clone()
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):
            # setup the number of features to use in discriminator
            self.criterion.set_ftr_num(self.ftr_num[stage])

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.flow_scheduler.update(curr_step, self.z_lrs[stage])
                self.flow_optim.zero_grad()

                self.frame3 = resample(self.frame2, self.flow)
                interp_result, flow_interpolate, warp_pred = self.interp_net(self.frame1, self.frame3)
                # calculate losses in the degradation space
                ftr_loss = self.criterion(self.ftr_net, interp_result, self.frame2)
                mse_loss = self.mse(interp_result, self.frame2)
                # nll corresponds to a negative log-likelihood loss
                l1_loss = F.l1_loss(interp_result, self.frame2)

                loss = ftr_loss * self.config['w_D_loss'][stage] + \
                    mse_loss * self.config['w_mse'][stage]
                loss.backward()

                self.flow_optim.step()
                loss_dict = {
                    'ftr_loss': ftr_loss,
                    'mse_loss': mse_loss / 4,
                    'l1_loss': l1_loss / 2
                }
                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        print(', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.4f' % (key, loss_dict[key]) for key in loss_dict]
                        ))
                    # save image sheet of the reconstruction process
                    save_imgs = torch.cat((save_imgs, self.frame3), dim=0)
                    torchvision.utils.save_image(
                        save_imgs.float(),
                        '%s/images_sheet/%s_%s.jpg' %
                        (self.config['exp_path'], self.img_name, self.mode),
                        nrow=int(save_imgs.size(0)**0.5),
                        normalize=True)

                if save_interval is not None:
                    if i == 0 or (i + 1) % save_interval[stage] == 0:
                        count += 1
                        save_path = '%s/images/%s' % (self.config['exp_path'],
                                                      self.img_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img_path = os.path.join(
                            save_path, '%s_%03d.jpg' % (self.img_name, count))
                        utils.save_img(x[0], img_path)

                # stop the reconstruction if the loss reaches a threshold
                if mse_loss.item() < self.config['stop_mse'] or ftr_loss.item(
                ) < self.config['stop_ftr']:
                    break

        # save images
        utils.save_img(
            self.gt[0], '%s/images/%s_%s_target.png' %
            (self.config['exp_path'], self.img_name, self.mode))
        return loss_dict

    def pre_process(self, image, target=True):
        if self.mode in ['SR', 'hybrid']:
            # apply downsampling, this part is the same as deep image prior
            if target:
                image_pil = utils.np_to_pil(
                    utils.torch_to_np((image.cpu() + 1) / 2))
                LR_size = [
                    image_pil.size[0] // self.factor,
                    image_pil.size[1] // self.factor
                ]
                img_LR_pil = image_pil.resize(LR_size, Image.ANTIALIAS)
                image = utils.np_to_torch(utils.pil_to_np(img_LR_pil)).cuda()
                image = image * 2 - 1
            else:
                image = self.downsampler((image + 1) / 2)
                image = image * 2 - 1
            # interpolate to the orginal resolution via bilinear interpolation
            image = F.interpolate(
                image, scale_factor=self.factor, mode='bilinear')
        n, _, h, w = image.size()
        if self.mode in ['colorization', 'hybrid']:
            # transform the image to gray-scale
            r = image[:, 0, :, :]
            g = image[:, 1, :, :]
            b = image[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            image = gray.view(n, 1, h, w).expand(n, 3, h, w)
        if self.mode in ['inpainting', 'hybrid']:
            # remove the center part of the image
            hole = min(h, w) // 3
            begin = (h - hole) // 2
            end = h - begin
            self.begin, self.end = begin, end
            mask = torch.ones(1, 1, h, w).cuda()
            mask[0, 0, begin:end, begin:end].zero_()
            image = image * mask
        return image

    def get_metrics(self, x):
        with torch.no_grad():
            l1_loss_origin = F.l1_loss(x, self.target_origin) / 2
            mse_loss_origin = self.mse(x, self.target_origin) / 4
            metrics = {
                'l1_loss_origin': l1_loss_origin,
                'mse_loss_origin': mse_loss_origin
            }
            # transfer to numpy array and scale to [0, 1]
            target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
            x_np = (x.detach().cpu().numpy()[0] + 1) / 2
            target_np = np.transpose(target_np, (1, 2, 0))
            x_np = np.transpose(x_np, (1, 2, 0))
            if self.mode == 'colorization':
                # combine the 'ab' dim of x with the 'L' dim of target image
                x_lab = color.rgb2lab(x_np)
                target_lab = color.rgb2lab(target_np)
                x_lab[:, :, 0] = target_lab[:, :, 0]
                x_np = color.lab2rgb(x_lab)
                x = torch.Tensor(np.transpose(x_np, (2, 0, 1))) * 2 - 1
                x = x.unsqueeze(0)
            elif self.mode == 'inpainting':
                # only use the inpainted area to calculate ssim and psnr
                x_np = x_np[self.begin:self.end, self.begin:self.end, :]
                target_np = target_np[self.begin:self.end,
                                      self.begin:self.end, :]
            ssim = compare_ssim(target_np, x_np, multichannel=True)
            psnr = compare_psnr(target_np, x_np)
            metrics['psnr'] = torch.Tensor([psnr]).cuda()
            metrics['ssim'] = torch.Tensor([ssim]).cuda()
            return metrics, x

    def jitter(self, x):
        save_imgs = x.clone().cpu()
        z_rand = self.z.clone()
        stds = [0.3, 0.5, 0.7]
        save_path = '%s/images/%s_jitter' % (self.config['exp_path'],
                                             self.img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with torch.no_grad():
            for std in stds:
                for i in range(30):
                    # add random noise to the latent vector
                    z_rand.normal_()
                    z = self.z + std * z_rand
                    x_jitter = self.G(z, self.G.shared(self.y))
                    utils.save_img(
                        x_jitter[0], '%s/std%.1f_%d.jpg' % (save_path, std, i))
                    save_imgs = torch.cat((save_imgs, x_jitter.cpu()), dim=0)

        torchvision.utils.save_image(
            save_imgs.float(),
            '%s/images_sheet/%s_jitters.jpg' %
            (self.config['exp_path'], self.img_name),
            nrow=int(save_imgs.size(0)**0.5),
            normalize=True)

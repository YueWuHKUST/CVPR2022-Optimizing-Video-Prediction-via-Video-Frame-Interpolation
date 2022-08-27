import numpy as np 
#import tensorflow as tf
import lpips_tf
import os
from PIL import Image
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
image_0 = tf.placeholder(tf.float32)
image_my = tf.placeholder(tf.float32)

lpips_my = lpips_tf.lpips(image_0, image_my, model='net-lin', net='alex')
msssim_my = tf.image.ssim_multiscale(image_0[0,...], image_my[0,...], max_val=1.0)
psnr_my = tf.image.psnr(image_0[0,...], image_my[0,...], max_val=1.0)
ssim_my = tf.image.ssim(image_0[0,...], image_my[0,...], max_val=1.0)

## Load GT images
davis_root = "/disk1/yue/data/videopred/DAVIS/DAVIS/JPEGImages/480p/"
val_list = '/disk1/yue/data/videopred/DAVIS/DAVIS/ImageSets/2017/val.txt'
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

pred_len = 5
num_vids = num_of_videos
#my_root = "/disk1/yue/code/iclr2022-prediction/baselines/deep-voxel-flow/davis_result/"
# my_root = "/disk1/yue/code/iclr2022-prediction/baselines/DYAN/DYAN_results/result_davis_480p_ucfweights_pyflow/"
my_root = "../cvpr2022_OMP_results/result_davis/"
my_videos = []
#for baseline
# for i in range(num_vids):
# 	sub_dir = my_root + "%04d/"%i
# 	images = []
# 	for k in range(pred_len):
# 		images.append(sub_dir + "pred_%04d.png"%k)
# 	my_videos.append(images)
for i in range(num_vids):
	sub_dir = my_root + "%s/"%files[i]
	images = []
	for k in range(pred_len):
		images.append(sub_dir + "pred_%04d.png"%k)
	my_videos.append(images)


ipips_score_mine = np.zeros(pred_len)
msssim_score_mine = np.zeros(pred_len)
psnr_score_mine = np.zeros(pred_len)
ssim_score_mine = np.zeros(pred_len)

width = 854
height = 480
for i in range(num_vids):
	#print("process ", i)
	for f in range(pred_len):
		true_image = np.expand_dims(np.array(Image.open(video[i][4 + f]).resize((width,height))), axis=0)/255
		my_image = np.expand_dims(np.array(Image.open(my_videos[i][f]).resize((width,height))), axis=0)/255
		#print(true_image.shape)
		#print(my_image.shape)

		ipips_m, msssim_m, psnr_m, ssim_m = sess.run([lpips_my, msssim_my, psnr_my, ssim_my], feed_dict={
					image_0: true_image, \
					image_my: my_image, 
			})
		#print("ssim shape", ssim.shape)
		ipips_score_mine[f] += ipips_m
		msssim_score_mine[f] += msssim_m
		psnr_score_mine[f] += psnr_m
		ssim_score_mine[f] += ssim_m

	for f in range(pred_len):
		print("iters", i)
		print("ipips my %d= "%f, ipips_score_mine[f]/(i+1), "ms-ssim %d= "%f, msssim_score_mine[f]/(i+1), \
				"psnr my %d= "%f, psnr_score_mine[f]/(i+1), "ssim my %d= "%f, ssim_score_mine[f]/(i+1))

for f in range(pred_len):
	print("ipips my %d= "%f, ipips_score_mine[f]/num_vids, "ms-ssim %d= "%f, msssim_score_mine[f]/num_vids, \
		"psnr my %d= "%f, psnr_score_mine[f]/num_vids, "ssim %d= "%f, ssim_score_mine[f]/num_vids)

for f in range(pred_len):
	print(ipips_score_mine[f]/num_vids, "\t", msssim_score_mine[f]/num_vids, "\t",\
		psnr_score_mine[f]/num_vids, "\t", ssim_score_mine[f]/num_vids)
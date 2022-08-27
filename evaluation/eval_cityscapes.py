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
distance_my = lpips_tf.lpips(image_0, image_my, model='net-lin', net='alex')
msssim_my = tf.image.ssim_multiscale(image_0[0,...], image_my[0,...], max_val=1.0)
psnr_my = tf.image.psnr(image_0[0,...], image_my[0,...], max_val=1.0)
ssim_my = tf.image.ssim(image_0[0,...], image_my[0,...], max_val=1.0)

## Load GT images
GT_root = "../dataset/Cityscapes/images_512x1024/val/"
city_dir = os.listdir(GT_root)
city_dir.sort()
true_videos = []
for i in range(len(city_dir)):
	frame_dir = GT_root + city_dir[i]
	frame_list = os.listdir(frame_dir)
	frame_list.sort()
	for j in range(len(frame_list)//30):
		image = []      
		for k in range(j*30, (j+1)*30):
			full_image_path = frame_dir + "/" + frame_list[k]
			assert os.path.isfile(full_image_path)
			image.append((full_image_path, city_dir[i], frame_list[k]))
		true_videos.append(image)

pred_len = 10

my_root = "../cvpr2022_OMP_results/result_cityscapes/"
num_vids = 500
my_videos = []
for i in range(num_vids):
	sub_dir = my_root + "%04d/"%i
	images = []
	for k in range(pred_len):
		images.append(sub_dir + "pred_%04d.png"%k)
	my_videos.append(images)

ipips_score_mine = np.zeros(pred_len)
msssim_score_mine = np.zeros(pred_len)
psnr_score_mine = np.zeros(pred_len)
ssim_score_mine = np.zeros(pred_len)


for i in range(num_vids):
	#print("process ", i)
	for f in range(pred_len):
		true_image = np.expand_dims(np.array(Image.open(true_videos[i][4 + f][0])), axis=0)/255
		my_image = np.expand_dims(np.array(Image.open(my_videos[i][f])), axis=0)/255
		ipips_m, msssim_m, psnr_m, ssim_m = sess.run([distance_my, msssim_my, psnr_my, ssim_my], feed_dict={
					image_0: true_image, \
					image_my: my_image, 
			})
		#print("ssim shape", ssim.shape)
		ipips_score_mine[f] += ipips_m
		msssim_score_mine[f] += msssim_m
		psnr_score_mine[f] += psnr_m
		ssim_score_mine[f] += ssim_m


	if i % 100 == 0:
		for f in range(pred_len):
			print("iters", i)
			print("ipips my %d= "%f, ipips_score_mine[f]/(i+1), "ms-ssim %d= "%f, msssim_score_mine[f]/(i+1), \
				 "psnr my %d= "%f, psnr_score_mine[f]/(i+1), "ssim my %d= "%f, ssim_score_mine[f]/(i+1))

for f in range(pred_len):
	print(ipips_score_mine[f]/num_vids, "\t", msssim_score_mine[f]/num_vids, "\t",\
		psnr_score_mine[f]/num_vids, "\t", ssim_score_mine[f]/num_vids)


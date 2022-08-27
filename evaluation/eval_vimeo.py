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

Vimeo_root = "../dataset/Vimeo/triplet/vimeo_triplet/sequences/"
output_dir = "./result_Vimeo_sintel/"
testlist = "../dataset/Vimeo/triplet/vimeo_triplet/tri_testlist.txt"
# readin files
testlist_f = open(testlist, 'r')
data = testlist_f.readlines()
testlist_f.close()
video = []

for i in range(len(data)-1):
    #print("data[i]", data[i], data[i].split("/"))
    seq_id, frame_id = data[i].split("/")
    frame_id = frame_id[:-1]#remove \n
    image_dir = Vimeo_root + seq_id + "/" + frame_id + "/"
    image_list = sorted(glob.glob(image_dir + "*.png"))
    video.append((image_list, seq_id, frame_id))
num_of_videos = len(video)
print("number of videos is ", num_of_videos)

pred_len = 1
num_vids = num_of_videos
#my_root = "/disk1/yue/code/iclr2022-prediction/internal_learning/result_Vimeo_sintel/"
# my_root = "/disk1/yue/code/iclr2022-prediction/baselines/deep-voxel-flow/vimeo_result/"
# /disk1/yue/code/iclr2022-prediction/cvpr2022_OMP_results/result_Vimeo
my_root = '../cvpr2022_OMP_results/result_Vimeo/'
my_videos = []
for i in range(num_vids):
	cnt_video = video[i]
	seq_id = cnt_video[1]
	frame_id = cnt_video[2]
	sub_dir = my_root + seq_id + "/" + frame_id + "/" 
	images = []
	for k in range(pred_len):
		images.append(sub_dir + "pred_%04d.png"%k)
	my_videos.append(images)


ipips_score_mine = np.zeros(pred_len)
msssim_score_mine = np.zeros(pred_len)
psnr_score_mine = np.zeros(pred_len)
ssim_score_mine = np.zeros(pred_len)
width = 448
height = 256
for i in range(num_vids):
	#print("process ", i)
	for f in range(pred_len):
		true_image = np.expand_dims(np.array(Image.open(video[i][0][2 + f]).resize((width,height))), axis=0)/255
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

	if i % 500 == 0:
		for f in range(pred_len):
			print("iters", i)
			print("ipips my %d= "%f, ipips_score_mine[f]/(i+1), "ms-ssim %d= "%f, msssim_score_mine[f]/(i+1), \
					"psnr my %d= "%f, psnr_score_mine[f]/(i+1), "ssim my %d= "%f, ssim_score_mine[f]/(i+1))

for f in range(pred_len):
	print(ipips_score_mine[f]/num_vids, "\t", msssim_score_mine[f]/num_vids, "\t",\
		psnr_score_mine[f]/num_vids, "\t", ssim_score_mine[f]/num_vids)
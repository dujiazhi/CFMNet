"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import models
from utils import batch_psnr,batch_ssim, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from ptflops import get_model_complexity_info

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test_ffdnet(**args):
	r"""Denoises an input image with FFDNet
	"""

	gray = args['gray']

	inputdir=args['input']
	outputdir=args['output']

	# Init logger
	logger = init_logger_ipol(args['logfile'])

	if not os.path.exists(outputdir):
		os.makedirs(outputdir)
	types = ('*.bmp', '*.png','*.jpg','*.tif')
	files = []
	for tp in types:
		files.extend(glob.glob(os.path.join(inputdir, tp)))
	files.sort()

	if not gray:
		in_ch = 3
		model_fn = 'logs/net.pth'

	else:
		in_ch = 1
		model_fn = 'logs/net.pth'

	model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
				model_fn)
	# Create model
	print('Loading model ...\n')
	#
	net = model.CFMNet(in_ch, in_ch, args.n_block)
	# Load saved weights
	if args['cuda']:
		state_dict = torch.load(model_fn)
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
	else:
		state_dict = torch.load(model_fn, map_location='cpu')
		# CPU mode: remove the DataParallel wrapper
		state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	# model.load_state_dict(state_dict['state_dict'])
	model.load_state_dict(state_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	if args['cuda']:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# Compute PSNR and log it
	if not gray:
		logger.info("### RGB denoising ###")
	else:
		logger.info("### Grayscale denoising ###")

	avg_noise_psnr=0
	avg_psnr=0
	avg_ssim=0
	avg_noise_ssim=0
	avg_time=0
	# stop_t=0
	# start_t=0
	cnt=len(files)
	for imagefile in files:
		(image_dir, imagename) = os.path.split(imagefile)
		(name, extension) = os.path.splitext(imagename)
		if not gray:
			imorig = cv2.imread(imagefile)
			imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		else:
			imorig = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
			imorig = np.expand_dims(imorig, 0)
		imorig = np.expand_dims(imorig, 0)

		# Handle odd sizes
		expanded_h = False
		expanded_w = False
		sh_im = imorig.shape
		if sh_im[2]%2 == 1:
			expanded_h = True
			imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

		if sh_im[3]%2 == 1:
			expanded_w = True
			imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
		ih,iw=imorig.shape[2:4]
		ih=ih//8*8
		iw=iw//8*8
		imorig=imorig[:,:,0:ih,0:iw]
		imorig = normalize(imorig)
		imorig = torch.Tensor(imorig)

		# Add noise
		if args['add_noise']:
			torch.manual_seed(0)
			# np.random.seed(0)
			noise = torch.FloatTensor(imorig.size()).\
					normal_(mean=0, std=args['noise_sigma'])
			imnoisy = imorig + noise
		else:
			imnoisy = imorig.clone()

		with torch.no_grad(): # PyTorch v0.4.0
			imorig, imnoisy = Variable(imorig.type(dtype)), \
						Variable(imnoisy.type(dtype))
			nsigma = Variable(\
				torch.FloatTensor([args['noise_sigma']]).type(dtype))


			# Measure runtime
			start_t = time.time()
			# Estimate noise and subtract it to the input image
			im_orig_estim = model(imnoisy, nsigma)
			stop_t = time.time()
			# outim = torch.clamp(imnoisy-im_orig_estim, 0., 1.)

			# im_noise_estim = model(imnoisy, nsigma)
			outim = torch.clamp(im_orig_estim, 0., 1.)
			# outim = torch.clamp(imnoisy-im_orig_estim, 0., 1.)



		if expanded_h:
			imorig = imorig[:, :, :-1, :]
			outim = outim[:, :, :-1, :]
			imnoisy = imnoisy[:, :, :-1, :]

		if expanded_w:
			imorig = imorig[:, :, :, :-1]
			outim = outim[:, :, :, :-1]
			imnoisy = imnoisy[:, :, :, :-1]

		if args['add_noise']:
			psnr = batch_psnr(outim, imorig, 1.)
			time1 = (stop_t - start_t)#*1000
			avg_psnr+=psnr
			avg_time+=time1
			psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
			avg_noise_psnr+=psnr_noisy
			ssim = batch_ssim(outim, imorig, 1.)
			avg_ssim += ssim
			ssim_noisy = batch_ssim(imnoisy, imorig, 1.)
			avg_noise_ssim += ssim_noisy

			logger.info("\tImagename:{0}, PSNR noisy {1:.2f}dB, PSNR denoised {2:.2f}dB".\
			            format(imagename,psnr_noisy,psnr))
		else:
			logger.info("\tNo noise was added, cannot compute PSNR")
		logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))


		# Save images'''
		if not args['dont_save_results']:
			noisyimg = variable_to_cv2_image(imnoisy)
			outimg = variable_to_cv2_image(outim)
			imorig = variable_to_cv2_image(imorig)
			# savepath为处理后文件保存的全路径
			noisyimgpath = os.path.join(outputdir, name+'_noisy'+extension)
			estimgpath = os.path.join(outputdir, name+'_CFMNet'+extension)
			cleanimgpath = os.path.join(outputdir, name+'_clean'+extension)

			cv2.imwrite(noisyimgpath, noisyimg)
			cv2.imwrite(cleanimgpath, imorig)

			# cv2.imwrite(estimgpath, outimg)
	avg_noise_psnr=0
	avg_psnr=0
	avg_ssim=0
	avg_noise_ssim=0
	avg_time=0
	for imagefile in files:
		(image_dir, imagename) = os.path.split(imagefile)
		(name, extension) = os.path.splitext(imagename)
		if not gray:
			imorig = cv2.imread(imagefile)
			imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
		else:
			imorig = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
			imorig = np.expand_dims(imorig, 0)
		imorig = np.expand_dims(imorig, 0)

		# Handle odd sizes
		expanded_h = False
		expanded_w = False
		sh_im = imorig.shape
		if sh_im[2]%2 == 1:
			expanded_h = True
			imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

		if sh_im[3]%2 == 1:
			expanded_w = True
			imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
		ih,iw=imorig.shape[2:4]
		ih=ih//8*8
		iw=iw//8*8
		imorig=imorig[:,:,0:ih,0:iw]
		imorig = normalize(imorig)
		imorig = torch.Tensor(imorig)

		# Add noise
		if args['add_noise']:
			torch.manual_seed(0)
			# np.random.seed(0)
			noise = torch.FloatTensor(imorig.size()).\
					normal_(mean=0, std=args['noise_sigma'])
			imnoisy = imorig + noise
		else:
			imnoisy = imorig.clone()

		with torch.no_grad(): # PyTorch v0.4.0
			imorig, imnoisy = Variable(imorig.type(dtype)), \
						Variable(imnoisy.type(dtype))
			nsigma = Variable(\
				torch.FloatTensor([args['noise_sigma']]).type(dtype))


			# Measure runtime
			start_t = time.time()
			# Estimate noise and subtract it to the input image
			im_orig_estim = model(imnoisy, nsigma)
			stop_t = time.time()
			# outim = torch.clamp(imnoisy-im_orig_estim, 0., 1.)

			# im_noise_estim = model(imnoisy, nsigma)
			outim = torch.clamp(im_orig_estim, 0., 1.)
			# outim = torch.clamp(imnoisy-im_orig_estim, 0., 1.)



		if expanded_h:
			imorig = imorig[:, :, :-1, :]
			outim = outim[:, :, :-1, :]
			imnoisy = imnoisy[:, :, :-1, :]

		if expanded_w:
			imorig = imorig[:, :, :, :-1]
			outim = outim[:, :, :, :-1]
			imnoisy = imnoisy[:, :, :, :-1]

		if args['add_noise']:
			psnr = batch_psnr(outim, imorig, 1.)
			time1 = (stop_t - start_t)#*1000
			avg_psnr+=psnr
			avg_time+=time1
			psnr_noisy = batch_psnr(imnoisy, imorig, 1.)
			avg_noise_psnr+=psnr_noisy
			ssim = batch_ssim(outim, imorig, 1.)
			avg_ssim += ssim
			ssim_noisy = batch_ssim(imnoisy, imorig, 1.)
			avg_noise_ssim += ssim_noisy

			logger.info("\tImagename:{0}, PSNR noisy {1:.2f}dB, PSNR denoised {2:.2f}dB".\
			            format(imagename,psnr_noisy,psnr))
		else:
			logger.info("\tNo noise was added, cannot compute PSNR")
		logger.info("\tRuntime {0:0.4f}s".format(stop_t-start_t))


		# Save images'''
		if not args['dont_save_results']:
			noisyimg = variable_to_cv2_image(imnoisy)
			outimg = variable_to_cv2_image(outim)
			# savepath为处理后文件保存的全路径
			noisyimgpath = os.path.join(outputdir, name+'_noisy'+extension)
			# estimgpath = os.path.join(outputdir, name+'_CFMNet'+extension)

			psnr_show = str(round(psnr, 2)).replace('.', '_')
			# noisyimgpath = os.path.join(outputdir, 'noisy_' + name + extension)
			estimgpath = os.path.join(outputdir, name + '_CFMNet_' + psnr_show + extension)

			# cv2.imwrite("./results/rgb/new/CFMNet_148026_" + str(int(args['noise_sigma_in'] * 255)) + ".png", outimg)

			# cv2.imwrite(estimgpath, outimg)
	logger.info("\t{0} has {1} images,average PSNR noisy {2:.2f}dB, average PSNR denoised {3:.2f}dB,\
				 time{4: .4f}"\
				.format(inputdir,cnt,avg_noise_psnr/cnt,avg_psnr/cnt,avg_time/cnt))
	logger.info("\t{0} has {1} images,average ssim noisy {2:.4f}dB, average ssim denoised {3:.4f}dB". \
				format(inputdir, cnt, avg_ssim / cnt, avg_noise_ssim / cnt))

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Model_Test")
	parser.add_argument("--gray",type=str,default='False',\
	                    help='input images is gray')
	parser.add_argument('--add_noise', type=str,default='True',\
	                    help='add noise to validate image(s)')
	parser.add_argument('--logfile', type=str, default='out', \
	                    help='logfile to log informations')
	parser.add_argument("--input", type=str, default="./imageset/Kodak24/", \
						help='path to input image(s)')
	parser.add_argument("--output", type=str, default="imresults/", \
						help='path to output image(s)')
	parser.add_argument("--noise_sigma", type=float, default=25, \
						help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', \
						help="don't save output images")
	parser.add_argument("--no_gpu", default=False, \
						help="run model on CPU")
	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# String to bool
	argspar.add_noise = (argspar.add_noise.lower() == 'true')
	argspar.gray = (argspar.gray.lower() == 'true')

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing  Model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_ffdnet(**vars(argspar))


"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.measure.simple_metrics import compare_psnr



def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   dynamic_range=data_range)
	return psnr/img_cpu.shape[0]


def variable_to_cv2_image(varim):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		varim: a torch.autograd.Variable
	"""
	nchannels = varim.size()[1]
	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		res = varim.data.cpu().numpy()[0]
		res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
		res = (res*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res




def init_logger_ipol():
	r"""Initializes a logging.Logger in order to log the results after
	testing a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler('out.txt', mode='w')
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger


def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)



def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict

def is_rgb(im):
	r""" Returns True if the image in im_path is an RGB image
	"""
	rgb = False
	if (len(im.shape) == 3):
		if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
			rgb = True
	
	
	return rgb

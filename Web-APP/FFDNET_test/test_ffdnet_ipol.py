import os
import argparse
import time
import numpy as np
import cv2
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from .models import FFDNet
from .utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from scipy.signal import convolve2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def estimate_noise(file):
    
    img = cv2.imread(file)
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = I.shape
    M = [[1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return int(sigma)


def test_ffdnet(input_path, out_path, cuda, noise_sigma):

        """Denoises an input image with FFDNet
        """
        # Normalize the nose sigma [0, 1] 
        noise_sigma /= 255.
        # Check if input exists and if it is RGB
        input = cv2.imread(input_path)
        rgb_den = is_rgb(input)
        if rgb_den:
            in_ch = 3
            model_fn = '../models/net_rgb.pth'
            imorig = input
            # from HxWxC to CxHxW, RGB image
            imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # from HxWxC to  CxHxW grayscale image (C=1)
            in_ch = 1
            model_fn = '../models/net_gray.pth'
            imorig = input
            imorig = np.expand_dims(imorig, 0)
        imorig = np.expand_dims(imorig, 0)
        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape
        if sh_im[2]%2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig, \
                    imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3]%2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig, \
                    imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        imorig = normalize(imorig)
        imorig = torch.Tensor(imorig)

        # Absolute path to model file
        model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
                    model_fn)

        # Create model
        net = FFDNet(num_input_channels=in_ch, test_mode=True)

        # Load saved weights
        if cuda:
            state_dict = torch.load(model_fn)
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(model_fn, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
            model = net
        model.load_state_dict(state_dict)
        # Sets the model in evaluation mode (e.g. it removes BN)
        model.eval()

        # Sets data type according to CPU or GPU modes
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        with torch.no_grad():
            imorig = Variable(imorig.type(dtype), volatile=True)
            nsigma = Variable(torch.FloatTensor([noise_sigma]).type(dtype), volatile=True)

        # Estimate noise and subtract it to the input image
        im_noise_estim = model(imorig, nsigma)
        outim = torch.clamp(imorig-im_noise_estim, 0., 1.)

        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]
            imorig = imorig[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
            imorig = imorig[:, :, :, :-1]

        if os.path.exists(out_path):
            os.remove(out_path)
        # Save output
        outimg = variable_to_cv2_image(outim)
        cv2.imwrite(out_path, outimg)
        
    
if __name__ == "__main__":
   # imorig = cv2.imread()
    noise_sigma = estimate_noise('/home/bakr/Desktop/woman.png')
    noise_sigma += (0.2 * noise_sigma)
    outimg = test_ffdnet ('/home/bakr/Desktop/woman.png','/home/bakr/Desktop/ffdnet_out.png', False, noise_sigma)
   # cv2.imwrite("ffdnet_out.png", outimg)

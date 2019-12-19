#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 21:32:04 2019

@author: marwan
"""
import os
from collections import OrderedDict
import torch
from .pcarn import Net 
import skimage.io as io
import torchvision.transforms as transforms
import cv2
import torchvision.utils as utils

def compute_image(LR_path,scale, SR_path):
    ckpt_path = '../models/PCARN-L1.pth'
    # Absolute path to model file
    ckpt_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), ckpt_path)
    lr = io.imread(LR_path)
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    lr = transform(lr)
    lr = lr.unsqueeze(0)

    kwargs = {
        "num_channels": 64,
        "groups": 1,
        "mobile": False,
        "scale": scale,
    }
    device = torch.device("cpu")
    lr = lr.to(device)
    net = Net(**kwargs).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    with torch.no_grad():
        SR = net(lr,scale).detach()
        
        
    utils.save_image(SR.squeeze(0), SR_path)

def main():

    compute_image("/media/bakr/Local\ Disk/computer/4th\ Year/Graduation\ project/code/Web-APP/PCARN_test/1/png",2,"/media/bakr/Local\ Disk/computer/4th\ Year/Graduation\ project/code/Web-APP/PCARN_test/1_o.png")

    

if __name__ == "__main__":
    main()


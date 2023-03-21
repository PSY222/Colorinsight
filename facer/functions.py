import argparse
import glob
import os
import os.path as osp
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.filters import gaussian

import facer
from model import BiSeNet


def get_rgb_codes(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    seg_probs = seg_probs.cpu() #if you are using GPU

    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    llip = tensor[:, :, 7]
    ulip = tensor[:,:,9]
    lips = llip+ulip
    binary_mask = (lips >= 0.5).astype(int)

    sample = cv2.imread(path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

    indices = np.argwhere(binary_mask)   #binary mask location extraction
    rgb_codes = img[indices[:, 0], indices[:, 1], :] #RGB color extraction by pixels
    return rgb_codes

def filter_lip_random(rgb_codes,randomNum=40):
    blue_condition = (rgb_codes[:, 2] <= 227)
    red_condition = (rgb_codes[:, 0] >= 97)
    filtered_rgb_codes = rgb_codes[blue_condition & red_condition]
    random_index = np.random.randint(0,filtered_rgb_codes.shape[0],randomNum)
    random_rgb_codes = filtered_rgb_codes[random_index]
    return random_rgb_codes


def calc_dis(rgb_codes):
    spring = [[253,183,169],[247,98,77],[186,33,33]]
    summer = [[243,184,202],[211,118,155],[147,70,105]]
    autum = [[210,124,110],[155,70,60],[97,16,28]]
    winter = [[237,223,227],[177,47,57],[98,14,37]]
  
    res = []
    for i in range(len(rgb_codes)):
      sp = np.inf
      su = np.inf
      au = np.inf
      win = np.inf
      for j in range(3):
        sp = min(sp, np.linalg.norm(rgb_codes[i] - spring[j]))
        su = min(su, np.linalg.norm(rgb_codes[i]- summer[j]))
        au = min(au, np.linalg.norm(rgb_codes[i] - autum[j]))
        win = min(win, np.linalg.norm(rgb_codes[i] - winter[j]))
    
      min_type = min(sp, su, au, win)
      if min_type == sp:
        ctype = "sp"
      elif min_type == su:
        ctype = "su"
      elif min_type == au:
        ctype = "au"
      elif min_type == win:
        ctype = "win"
    
      res.append(ctype)
    return res


def save_skin_mask(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)  # image: 1 x 3 x h x w
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

    with torch.inference_mode():
      faces = face_detector(image)

    image = facer.hwc2bchw(facer.read_hwc(img_path)).to(device=device)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
      faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    seg_probs = seg_probs.cpu() #if you are using GPU
    tensor = seg_probs.permute(0, 2, 3, 1)
    tensor = tensor.squeeze().numpy()

    face_skin = tensor[:, :, 1]
    binary_mask = (face_skin >= 0.5).astype(int)

    sample = cv2.imread(img_path)
    img = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    masked_image = np.zeros_like(img) 
    try: 
      masked_image[binary_mask == 1] = img[binary_mask == 1] 
      masked_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB)
      cv2.imwrite("temp.jpg" , masked_image)
    except:
      print("error occurred")
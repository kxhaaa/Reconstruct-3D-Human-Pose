import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
#import argparse
import json

#from models import hmr, SMPL
from utils.imutils import crop
# utils.renderer import Renderer
#import config
import constants



def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bboxs = np.array(json.load(f)['bbox']).astype(np.float32)
    centers = []
    scales = []
    for bbox in bboxs:
        ul_corner = bbox[:2]
        center = ul_corner + 0.5 * bbox[2:]
        width = max(bbox[2], bbox[3])
        scale = width / 200.0
        centers.append(center)
        scales.append(scale)
    # make sure the bounding box is rectangular
    return centers, scales
#we hope this function will return several img correspond to different bounding box in one img
def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    oimg = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    centers, scales = bbox_from_json(bbox_file)
    imgs = []
    norm_imgs = []
    for i in range(len(centers)):
        img = crop(oimg, centers[i], scales[i], (input_res, input_res))
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2,0,1)
        norm_img = normalize_img(img.clone())[None]
        imgs.append(img)
        norm_imgs.append(norm_img)
    return imgs, norm_imgs

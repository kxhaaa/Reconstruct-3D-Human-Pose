#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:31:02 2021

@author: kong
"""
import torch
from models import hmr, SMPL
import config
from smplify import SMPLify
import argparse
from utils.renderer import Renderer
import constants
#import OpenGL
import numpy as np
import cv2
from predict_preprocess import process_image

# =============================================================================
# 1, set the model
# =============================================================================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = hmr(config.SMPL_MEAN_PARAMS,pretrained=True).to(device)

checkpoint = torch.load('data/model_checkpoint.pt')

model.load_state_dict(checkpoint['model'], strict=False)

# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)

model.eval()

renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

# =============================================================================
# 2, load the data
# =============================================================================

imgs, norm_imgs = process_image('test_img1.jpg',
                              bbox_file = 'bbox1.json', openpose_file=None,
                              input_res=constants.IMG_RES)

# img_name = 'FudanPed00012.png'
# img, norm_img = process_image('examples/'+img_name,
#                               bbox_file = None, openpose_file=None,
#                               input_res=constants.IMG_RES)

# =============================================================================
# 3, prediction
# =============================================================================
for i in range(len(imgs)):
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_imgs[i].to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = imgs[i].permute(1,2,0).cpu().numpy()

    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center

    # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

    outfile = 'out_file/'

    # Save reconstructions
    cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
    cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])



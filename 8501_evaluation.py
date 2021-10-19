#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:50:48 2021

@author: kong
"""
import torch
from torch.utils.data import DataLoader
import numpy as np

import config
import constants
from models import hmr, SMPL
from datasets.rewrite_basedataset import BaseDataset
from utils.pose_utils import reconstruction_error


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_name = '3dpw'
dataset = BaseDataset(dataset_name, is_train=False)

model = hmr(config.SMPL_MEAN_PARAMS,pretrained=True).to(device)
model.eval()

checkpoint = torch.load('data/model_checkpoint.pt')

model.load_state_dict(checkpoint['model'], strict=False)

# Load SMPL model
smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                    create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR,
                   gender='female',
                   create_transl=False).to(device)

J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()


batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)


# =============================================================================
# Parameters initialization
# =============================================================================
# Pose metrics
# MPJPE and Reconstruction error for the non-parametric and parametric shapes
mpjpe = np.zeros(len(dataset))
recon_err = np.zeros(len(dataset))
mpjpe_smpl = np.zeros(len(dataset))
recon_err_smpl = np.zeros(len(dataset))

# Shape metrics
# Mean per-vertex error
shape_err = np.zeros(len(dataset))
shape_err_smpl = np.zeros(len(dataset))

# Mask and part metrics
# Accuracy
accuracy = 0.
parts_accuracy = 0.
# True positive, false positive and false negative
tp = np.zeros((2,1))
fp = np.zeros((2,1))
fn = np.zeros((2,1))
parts_tp = np.zeros((7,1))
parts_fp = np.zeros((7,1))
parts_fn = np.zeros((7,1))
# Pixel count accumulators
pixel_count = 0
parts_pixel_count = 0

# Store SMPL parameters
smpl_pose = np.zeros((len(dataset), 72))
smpl_betas = np.zeros((len(dataset), 10))
smpl_camera = np.zeros((len(dataset), 3))
pred_joints = np.zeros((len(dataset), 17, 3))

eval_pose = False
eval_masks = False
eval_parts = False


# =============================================================================
# evaluation
# =============================================================================
if dataset_name == '3dpw':
    eval_pose = True

elif dataset_name == 'lsp':  # not accessible yet, need neural_renderer package
    eval_masks = True
    eval_parts = True
    annot_path = config.DATASET_FOLDERS['upi-s1h']

joint_mapper_h36m = constants.H36M_TO_J14
joint_mapper_gt = constants.J24_TO_J14

log_freq = 2
# Iterate over the entire dataset
for step, batch in enumerate(data_loader):
    # Get ground truth annotations from the batch
    gt_pose = batch['pose'].to(device)
    gt_betas = batch['betas'].to(device)
    gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
    images = batch['img'].to(device)
    gender = batch['gender'].to(device)
    curr_batch_size = images.shape[0]
    
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(images)
        pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        
    # 3D pose evaluation
    if eval_pose:
        # Regressor broadcasting
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
        # Get 14 ground truth joints
        if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
        # For 3DPW get the 14 common joints from the rendered shape
        else:
            gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
            gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
            gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

        # Reconstuction_error
        r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error
        
    if step % log_freq == log_freq - 1:
        if eval_pose:
            print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
            print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
            print()


# Print final results during evaluation
print('*** Final Results ***')
print()
if eval_pose:
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
    print()






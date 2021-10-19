#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:07:35 2021

@author: kong
"""
from train.fits_dict import FitsDict
from utils import TrainOptions

from torchgeometry import rotation_matrix_to_angle_axis
import torch.nn as nn
from utils import TrainOptions
from smplify import SMPLify
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import numpy as np
from utils.geometry import estimate_translation, perspective_projection, batch_rodrigues
import cv2
from os.path import join
from models import hmr, SMPL
import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils import CheckpointDataLoader
from datasets.rewrite_basedataset import BaseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Lossfn
# =============================================================================
criterion_regr = nn.MSELoss().to(device)
criterion_keypoints = nn.MSELoss(reduction='none').to(device)
criterion_shape = nn.L1Loss().to(device)

def smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(device)
        return loss_regr_pose, loss_regr_betas

def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss  

def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(device)

def shape_loss(pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(device)


# =============================================================================
# 1, dataset
# =============================================================================
lsp_dataset = BaseDataset('lsp-orig')
options = TrainOptions().parse_args()
fits_dict = FitsDict(options, lsp_dataset)

# =============================================================================
# 2, models
# =============================================================================
batch_size=16
focal_len=constants.FOCAL_LENGTH
model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(device)
smpl = SMPL(config.SMPL_MODEL_DIR, batch_size, create_transl=False).to(device)
smplify = SMPLify(step_size=1e-2, batch_size=batch_size, num_iters=100, focal_length = focal_len)

# =============================================================================
# 3, training pipline
# =============================================================================
dataloader = DataLoader(lsp_dataset, batch_size)
optimizer = torch.optim.Adam(params=model.parameters(),lr=options.lr)
epoches = 2
losses = []

print('start training')
for epoch in range(epoches):
    for i, batch in enumerate(dataloader):
        print('-'*60)
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        input_batch = batch
        
        # # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'] # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'] # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]
        
        ##1, process of the ground truth
        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices
        
        # Get current best fits from the dictionary
        opt_pose, opt_betas = fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints
        
        
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * 224 * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        
        
        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=constants.FOCAL_LENGTH, img_size=224)
        
        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=constants.FOCAL_LENGTH, img_size=224)
        
        
        opt_joint_loss = smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                                0.5 * options.img_res * torch.ones(batch_size, 2, device=device),
                                                                gt_keypoints_2d_orig).mean(dim=-1)


        #### 2, HMR parts
        pred_rotmat, pred_betas, pred_camera = model(images)    # images should be 64,3,224,224

        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        
        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                          pred_camera[:,2],
                                          2*constants.FOCAL_LENGTH/(224 * pred_camera[:,0] +1e-9)],dim=-1)
        
        
        camera_center = torch.zeros(batch_size, 2, device=device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                            translation=pred_cam_t,
                                                            focal_length=constants.FOCAL_LENGTH,
                                                            camera_center=camera_center)
        # Normalize keypoints to [-1,1]        
        pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)
        
        
        #### 3, SMPLify parts
        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                        device=device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom)
        pred_pose = pred_pose.contiguous().view(batch_size, -1)
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
        pred_pose[torch.isnan(pred_pose)] = 0.0
        
        # Run SMPLify optimization starting from the network prediction
        new_opt_vertices, new_opt_joints,\
                    new_opt_pose, new_opt_betas,\
                    new_opt_cam_t, new_opt_joint_loss = smplify(
                                                pred_pose.detach(), pred_betas.detach(),
                                                pred_cam_t.detach(),
                                                0.5 * 224 * torch.ones(batch_size, 2, device=device),
                                                gt_keypoints_2d_orig)
        new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
        
        
        # Will update the dictionary for the examples where the new loss is less than the current one
        update = (new_opt_joint_loss < opt_joint_loss)
                    
        opt_joint_loss[update] = new_opt_joint_loss[update]
        opt_vertices[update, :] = new_opt_vertices[update, :]
        opt_joints[update, :] = new_opt_joints[update, :]
        opt_pose[update, :] = new_opt_pose[update, :]
        opt_betas[update, :] = new_opt_betas[update, :]
        opt_cam_t[update, :] = new_opt_cam_t[update, :]
        fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())
        
        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.
        
        # Replace the optimized parameters with the ground truth parameters, if available
        if has_smpl.sum() == 0:
            opt_vertices[:, :, :] = gt_vertices[:, :, :]
            opt_cam_t[:, :] = gt_cam_t[:, :]
            opt_joints[:, :, :] = gt_model_joints[:, :, :]
            opt_pose[:, :] = gt_pose[:, :]
            opt_betas[:, :] = gt_betas[:, :]
        
        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < 100).to(device)
        
        opt_keypoints_2d = perspective_projection(opt_joints,
                                                          rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                          translation=opt_cam_t,
                                                          focal_length=focal_len,
                                                          camera_center=camera_center)
        
        opt_keypoints_2d = opt_keypoints_2d / (224 / 2.)
        
        
        ### 4, Loss
        loss_regr_pose, loss_regr_betas = smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                                    options.openpose_train_weight,
                                                    options.gt_train_weight)
        
        # Compute 3D keypoint loss
        loss_keypoints_3d = keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)
        
        # Per-vertex loss for the shape
        loss_shape = shape_loss(pred_vertices, opt_vertices, valid_fit)
        
        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = options.shape_loss_weight * loss_shape +\
                       options.keypoint_loss_weight * loss_keypoints +\
                       options.keypoint_loss_weight * loss_keypoints_3d +\
                       options.pose_loss_weight * loss_regr_pose + options.beta_loss_weight * loss_regr_betas +\
                       ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        loss *= 60
        losses.append(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch %d: batch %d|%d, loss is %.4f'%(epoch, i, len(dataloader), loss.item()))








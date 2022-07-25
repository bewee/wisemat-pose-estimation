import torch
import numpy as np
from .constants import constants

def _convert_relative_to_absolute(p):
    original_shape = p.shape
    p = torch.clone(p).reshape(-1, 2)
    p[:,0] *= constants.MATTRESS_WIDTH_CM
    p[:,1] *= constants.MATTRESS_HEIGHT_CM
    return p.reshape(original_shape)
    
def _euclidean_dist(p1, p2):
    original_shape = p1.shape
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    diff = torch.square(p2 - p1)
    dist = torch.sqrt(diff[:,0] + diff[:,1])
    if original_shape[-1] == 2:
        return dist.reshape(original_shape[:-1])
    return dist

def per_joint_position_errors(y_hat, y, joints=None):
    if joints == None:
        joints = constants.JOINTS
    y_hat = _convert_relative_to_absolute(y_hat.reshape(-1, joints, 2))
    y = _convert_relative_to_absolute(y.reshape(-1, joints, 2))
    return torch.transpose(_euclidean_dist(y_hat, y), 0, 1)

def mpjpe(y_hat, y):
    return torch.mean(per_joint_position_errors(y_hat, y))

def mean_squared_per_joint_position_error(y_hat, y):
    y_hat = _convert_relative_to_absolute(y_hat.reshape(-1, 2))
    y = _convert_relative_to_absolute(y.reshape(-1, 2))
    diff = torch.square(y_hat - y)
    return torch.mean(diff[:,0] + diff[:,1])

def per_joint_mean_position_errors(y_hat, y):
    return per_joint_position_errors(y_hat, y).mean(axis=1)

def _part_correct_percentage(joint1, joint2, joint1_hat, joint2_hat, at = 0.5):
    limb_length = _euclidean_dist(joint1, joint2)
    joint1_dist = _euclidean_dist(joint1, joint1_hat)
    joint2_dist = _euclidean_dist(joint2, joint2_hat)
    
    c = (joint1_dist + joint2_dist) / 2 < at * limb_length
    return (torch.sum(c) / c.shape[0]).cpu().numpy()

def per_part_correct_percentage(y_hat, y, at = 0.5):
    if constants.SKELETON_FORMAT == 'common':
        y_hat = _convert_relative_to_absolute(y_hat.reshape(-1, constants.JOINTS, 2))
        y = _convert_relative_to_absolute(y.reshape(-1, constants.JOINTS, 2))
        return torch.Tensor(np.asarray([
            _part_correct_percentage(y[:,0,:],  y[:,1,:],  y_hat[:,0,:],  y_hat[:,1,:],  at), # right lower leg
            _part_correct_percentage(y[:,1,:],  y[:,2,:],  y_hat[:,1,:],  y_hat[:,2,:],  at), # right upper leg
            _part_correct_percentage(y[:,3,:],  y[:,4,:],  y_hat[:,3,:],  y_hat[:,4,:],  at), # left upper leg
            _part_correct_percentage(y[:,4,:],  y[:,5,:],  y_hat[:,4,:],  y_hat[:,5,:],  at), # left lower leg
            _part_correct_percentage(y[:,6,:],  y[:,7,:],  y_hat[:,6,:],  y_hat[:,7,:],  at), # right forearm
            _part_correct_percentage(y[:,7,:],  y[:,8,:],  y_hat[:,7,:],  y_hat[:,8,:],  at), # right upper arm
            _part_correct_percentage(y[:,9,:],  y[:,10,:], y_hat[:,9,:],  y_hat[:,10,:], at), # left upper arm
            _part_correct_percentage(y[:,10,:], y[:,11,:], y_hat[:,10,:], y_hat[:,11,:], at), # left forearm
        ], dtype=np.float32))
    if constants.SKELETON_FORMAT == 'arms-omitted':
        y_hat = _convert_relative_to_absolute(y_hat.reshape(-1, constants.JOINTS, 2))
        y = _convert_relative_to_absolute(y.reshape(-1, constants.JOINTS, 2))
        return torch.Tensor(np.asarray([
            _part_correct_percentage(y[:,0,:],  y[:,1,:],  y_hat[:,0,:],  y_hat[:,1,:],  at), # right lower leg
            _part_correct_percentage(y[:,1,:],  y[:,2,:],  y_hat[:,1,:],  y_hat[:,2,:],  at), # right upper leg
            _part_correct_percentage(y[:,3,:],  y[:,4,:],  y_hat[:,3,:],  y_hat[:,4,:],  at), # left upper leg
            _part_correct_percentage(y[:,4,:],  y[:,5,:],  y_hat[:,4,:],  y_hat[:,5,:],  at), # left lower leg
        ], dtype=np.float32))
    else:
        raise Exception("Not yet supported")

def pcp(y_hat, y, at = 0.5):
    return torch.mean(per_part_correct_percentage(y_hat, y, at)) 
    
def _keypoint_correct_percentage(joint, joint_hat, reference_length, at = 0.2):
    diff = _euclidean_dist(joint, joint_hat)
    c = diff < at * reference_length
    return torch.sum(c) / c.shape[0]

def per_keypoint_correct_percentage(y_hat, y, at = 0.2):
    y_hat = _convert_relative_to_absolute(y_hat.reshape(-1, constants.JOINTS, 2))
    y = _convert_relative_to_absolute(y.reshape(-1, constants.JOINTS, 2))

    if constants.SKELETON_FORMAT == 'common':
        torso_diameter = _euclidean_dist(y[:,12,:], (y[:,2,:] + y[:,3,:])/2)
        result = torch.empty(constants.JOINTS)
        for i in range(constants.JOINTS):
            result[i] = _keypoint_correct_percentage(y[:,i,:], y_hat[:,i,:], torso_diameter, at)
        return result
    if constants.SKELETON_FORMAT == 'arms-omitted':
        torso_diameter = _euclidean_dist(y[:,8,:], (y[:,2,:] + y[:,3,:])/2)
        result = torch.empty(constants.JOINTS)
        for i in range(constants.JOINTS):
            result[i] = _keypoint_correct_percentage(y[:,i,:], y_hat[:,i,:], torso_diameter, at)
        return result
    else:
        raise Exception("Not yet supported")

def pck(y_hat, y, at=0.2):
    return torch.mean(per_keypoint_correct_percentage(y_hat, y, at))

def mpjpe_arms_omitted(y_hat, y):
    if constants.SKELETON_FORMAT == 'common':
        y_hat = y_hat.reshape(-1, 13, 2)
        y = y.reshape(-1, 13, 2)
        y_hat = torch.cat((y_hat[:,:6,:], y_hat[:,8:10,:], y_hat[:,12:,:]), dim=1)
        y = torch.cat((y[:,:6,:], y[:,8:10,:], y[:,12:,:]), dim=1)
        return torch.mean(per_joint_position_errors(y_hat, y, joints=9))
    else:
        raise Exception("Not yet supported")

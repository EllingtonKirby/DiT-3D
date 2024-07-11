import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import json
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from modules.three_d_helpers import cartesian_to_cylindrical
import open3d as o3d
from modules.class_mapping import class_mapping

class NuscenesObjectsSet(Dataset):
    def __init__(self, 
                data_dir,
                split, 
                points_per_object=None, 
                volume_expansion=1., 
                recenter=True, 
                align_objects=False, 
                relative_angles=False, 
                stacking_type='duplicate',
                class_conditional=False,
                normalize_points=False,
                ):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]

        self.nr_data = len(self.data_index)
        self.points_per_object = points_per_object
        self.volume_expansion = volume_expansion
        self.do_recenter = recenter
        self.align_objects = align_objects
        self.relative_angles = relative_angles
        self.stacking_type = stacking_type
        self.class_conditional = class_conditional
        self.normalize_points = normalize_points

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = object_json['class']
        points = np.fromfile(object_json['lidar_data_filepath'], dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        center = np.array(object_json['center'])
        size = np.array(object_json['size'])
        rotation_real = np.array(object_json['rotation_real'])
        rotation_imaginary = np.array(object_json['rotation_imaginary'])

        orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
        box = Box(center=center, size=size, orientation=orientation)
        
        points_from_object = points_in_box(box, points=points[:,:3].T, wlh_factor=self.volume_expansion)
        object_points = torch.from_numpy(points[points_from_object])[:,:3]
        
        num_points = object_points.shape[0]
        if self.points_per_object > 0 and self.stacking_type != 'max':
            if object_points.shape[0] > self.points_per_object:
                pcd_object = o3d.geometry.PointCloud()
                pcd_object.points = o3d.utility.Vector3dVector(object_points)
                pcd_object = pcd_object.farthest_point_down_sample(self.points_per_object)
                object_points = torch.tensor(np.array(pcd_object.points))
                padding_mask = torch.ones((object_points.shape[0]))
            else:
                if self.stacking_type == 'pad':
                    padded_points = torch.zeros((self.points_per_object, object_points.shape[1]))
                    padded_points[:object_points.shape[0]] = object_points
                    object_points = padded_points
                    padding_mask = torch.zeros((object_points.shape[0]))
                    padding_mask[:num_points] = 1
                elif self.stacking_type == 'duplicate':
                    concat_part = int(np.ceil(self.points_per_object / object_points.shape[0]) )
                    object_points = object_points.repeat(concat_part, 1)
                    object_points = object_points[torch.randperm(object_points.shape[0])][:self.points_per_object]
                    padding_mask = torch.ones((object_points.shape[0]))
        else:
            padding_mask = torch.zeros((object_points.shape[0]))
        
        ring_indexes = torch.zeros_like(object_points)
        if self.do_recenter:
            object_points -= center
        
        center = cartesian_to_cylindrical(center[None,:])[0]
        yaw = orientation.yaw_pitch_roll[0]
        
        if self.align_objects:
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            object_points = np.dot(object_points, rotation_matrix.T)

        if self.relative_angles:
            center[0] -= yaw

        if self.class_conditional:
            class_label = torch.tensor(class_mapping[class_name])
        else:
            class_label = torch.ones((1)).int()

        if self.normalize_points:
            # Determine the axis with the largest range in the point cloud
            longest_axis_range = np.max(object_points, axis=0) - np.min(object_points, axis=0)
            longest_axis = np.argmax(longest_axis_range).item()

            # Find the min and max values along this longest axis
            min_val = np.min(object_points[:, longest_axis])
            max_val = np.max(object_points[:, longest_axis])

            # Normalize the point cloud based on the longest axis
            object_points = (object_points - min_val) / (max_val - min_val)

        return [object_points, center, torch.from_numpy(size), yaw, num_points, ring_indexes, class_label, padding_mask]
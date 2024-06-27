import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import glob

class ShapeNetObjectsSet(Dataset):
    def __init__(self, data_dir, split, points_per_object):
        super().__init__()
        self.data_dir = f'{data_dir}/{split}'
        self.files = glob.glob(f'{self.data_dir}/**.npy')
        self.nr_data = len(self.files)
        self.points_per_object = points_per_object

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_points = np.load(self.files[index])

        if self.points_per_object > 0:
            object_points = object_points[torch.randperm(object_points.shape[0])][:self.points_per_object]
        
        num_points = object_points.shape[0]
        rotation_matrix_pitch = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        rotated_point_cloud = object_points.dot(rotation_matrix_pitch.T)
        size = np.zeros(3)
        center = np.zeros(3)
        orientation = np.zeros(1)
        ring_indexes = np.zeros_like(object_points)
        class_name = 'vehicle.motorcycle'


        return [rotated_point_cloud, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name]
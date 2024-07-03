import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets.dataloader_nuscenes import NuscenesObjectsSet
import warnings
import numpy as np

warnings.filterwarnings('ignore')

__all__ = ['NuscenesObjectsDataModule']

class NuscenesObjectsDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = NuscenesObjectCollator(self.cfg['data']['stacking_type'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train', 
                points_per_object=self.cfg['data']['points_per_object'],
                align_objects=self.cfg['data']['align_objects'],
                relative_angles=self.cfg['model']['relative_angles'],
                stacking_type=self.cfg['data']['stacking_type']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=self.cfg['data']['stacking_type'] != 'max',
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = NuscenesObjectCollator(self.cfg['data']['stacking_type'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
                points_per_object=self.cfg['data']['points_per_object'],
                align_objects=self.cfg['data']['align_objects'],
                relative_angles=self.cfg['model']['relative_angles'],
                stacking_type=self.cfg['data']['stacking_type']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = NuscenesObjectCollator(self.cfg['data']['stacking_type'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val', 
                points_per_object=self.cfg['data']['points_per_object'],
                align_objects=self.cfg['data']['align_objects'],
                relative_angles=self.cfg['model']['relative_angles'],
                stacking_type=self.cfg['data']['stacking_type']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class NuscenesObjectCollator:
    def __init__(self, stacking_type):
        self.max_stack = stacking_type == 'max'
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        num_points_tensor = torch.Tensor(batch[4])
        # cumulative_indices = torch.cumsum(num_points_tensor, dim=0).long()
        # batch_indices[cumulative_indices-1] = 1
        # batch_indices = batch_indices.cumsum(0).long()
        # batch_indices[-1] = batch_indices[-2]
        if self.max_stack:
            pcd_object = batch[0]
            max_points = num_points_tensor.max().int().item()
            padded_point_clouds = np.zeros((len(pcd_object), max_points, 3))
            padding_mask = np.zeros((len(pcd_object), max_points))
            for i, pc in enumerate(pcd_object):
                padded_point_clouds[i, :pc.shape[0], :] = pc
                padding_mask [i, :pc.shape[0]] = 1

            pcd_object = torch.from_numpy(padded_point_clouds).float()
            padding_mask = torch.from_numpy(padding_mask).float()
        else:
            pcd_object = torch.from_numpy(np.stack(batch[0]))
            padding_mask = torch.from_numpy(np.stack(batch[7]))
        pcd_object = pcd_object.permute(0,2,1)
        batch_indices = torch.zeros(len(pcd_object))

        # class_mapping = torch.tensor([data_map.class_mapping[class_name] for class_name in batch[6]]).reshape(-1, 1)
        # num_classes = max(data_map.class_mapping.values()) + 1
        # class_mapping = torch.nn.functional.one_hot(class_mapping, num_classes=num_classes)

        return {'pcd_object': pcd_object, 
            'center':  torch.from_numpy(np.vstack(batch[1])).float(),
            'size':  torch.from_numpy(np.vstack(batch[2])).float(),
            'orientation': torch.from_numpy(np.vstack(batch[3])).float(),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'ring_indexes': torch.vstack(batch[5]),
            'class': torch.ones((pcd_object.shape[0], 1)).long(),
            'padding_mask': padding_mask
        }

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
}


from datasets.dataset_nuscenes import NuscenesObjectsDataModule
from datasets.dataset_shapenet import ShapeNetObjectsDataModule

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
    'shapenet': ShapeNetObjectsDataModule
}

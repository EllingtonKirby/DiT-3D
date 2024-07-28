import click
import os
from os.path import join, dirname, abspath
import numpy as np
import torch
import yaml
from tqdm import tqdm
from datasets.dataset_mapper import dataloaders
from models.models_dit3d_dropout import DiT3D_Diffuser
from diffusers import DPMSolverMultistepScheduler
from modules.three_d_helpers import cylindrical_to_cartesian, angle_add

def realign_pointclouds_to_scan(x_gen, orientation, center, aligned_angle):
    cos_yaw = np.cos(orientation)
    sin_yaw = np.sin(orientation)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    x_gen[:, :3] = np.dot(x_gen[:, :3], rotation_matrix.T)
    new_center = center.copy()
    new_center[0] = angle_add(aligned_angle, orientation)
    new_center = cylindrical_to_cartesian(new_center[None, :]).squeeze(0)
    x_gen[:, :3] += new_center
    x_gen[:, 3] = (x_gen[:, 3] + 1) * 15.  # rescale intensity to 0-30
    return x_gen

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--num_instances',
              '-n',
              type=int,
              help='number of examples to generate from each sample in the dataset',
              default=1) # doubling the number of instances
@click.option('--split',
              '-s',
              type=str,
              help='train or val split',
              default='train')
@click.option('--rootdir',
              '-r',
              type=str,
              default=None)
def main(config, weights, num_instances, split, rootdir):
    cfg = yaml.safe_load(open(config))
    experiment_dir = cfg['experiment']['id']
    model = DiT3D_Diffuser.load_from_checkpoint(weights, hparams=cfg)
    dl = cfg['data']['dataloader']
    train_dl = dataloaders[dl](cfg).train_dataloader()
    val_dl =   dataloaders[dl](cfg).val_dataloader()
    dl = train_dl if split == 'train' else val_dl
    model.eval()

    for _, batch in tqdm(enumerate(dl)):
        with torch.no_grad():
            x_object = batch['pcd_object']
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            padding_mask = batch['padding_mask'].cuda()
            annotation_tokens = batch['tokens']
            if num_instances == 1:
                linspace_ring = x_center[:, 0].unsqueeze(0) # recreate exact example
            else:
                random_angles = torch.rand((x_object.shape[0])) * 2 * torch.pi - torch.pi
                linspace_ring = np.linspace(start=random_angles, stop=random_angles + 2 * np.pi, num=num_instances, endpoint=False)
                linspace_ring = (linspace_ring + np.pi) % (2 * np.pi) - np.pi
                linspace_ring = torch.from_numpy(linspace_ring)

            for i in range(num_instances):
                model.dpm_scheduler = DPMSolverMultistepScheduler(
                        num_train_timesteps=model.t_steps,
                        beta_start=model.hparams['diff']['beta_start'],
                        beta_end=model.hparams['diff']['beta_end'],
                        beta_schedule='linear',
                        algorithm_type='sde-dpmsolver++',
                        solver_order=2,
                    )
                model.dpm_scheduler.set_timesteps(model.s_steps)
                model.scheduler_to_cuda()
                x_t = torch.randn(x_object.shape, device=model.device)
                if cfg['model']['cyclic_conditions'] > 0:
                    if cfg['model']['relative_angles'] == True:
                        x_cond = torch.cat((x_center, x_size),-1)
                    else:
                        x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
                else:
                    x_cond = torch.hstack((x_center, x_size, x_orientation))
                x_cond[:, 0] = linspace_ring[i]
                x_cond = x_cond.cuda()
                x_generated = model.p_sample_loop(x_t, batch['class'], x_cond, padding_mask[:, None, :]).permute(0,2,1).cpu().detach().numpy()
                x_object = x_object.permute(0,2,1).cpu().detach().numpy()
                for j in range(x_generated.shape[0]):
                    mask = padding_mask[j].cpu().detach().numpy().astype(np.bool_)
                    x_gen = x_generated[j][mask]
                    x_org = x_object[j][mask]
                    center = x_center[j].cpu().detach().numpy()
                    size = x_size[j].cpu().detach().numpy()
                    orientation = x_orientation[j].cpu().detach().numpy()

                    x_gen = realign_pointclouds_to_scan(x_gen, orientation.item(), center, linspace_ring[i, j])
                    x_org = realign_pointclouds_to_scan(x_org, orientation.item(), center, linspace_ring[i, j])

                    conditions = np.concatenate((center, size, orientation))
                    token = annotation_tokens[j]
                    os.makedirs(f'{rootdir}/{experiment_dir}/{split}/{token}', exist_ok=True)
                    np.savetxt(f'{rootdir}/{experiment_dir}/{split}/{token}/generated_{i}.txt', x_gen)
                    np.savetxt(f'{rootdir}/{experiment_dir}/{split}/{token}/original_{i}.txt', x_org)
                    np.savetxt(f'{rootdir}/{experiment_dir}/{split}/{token}/conditions_{i}.txt', conditions)


if __name__ == "__main__":
    main()

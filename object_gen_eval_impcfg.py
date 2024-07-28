import os
from pytorch_lightning import LightningDataModule
import yaml
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from diffusers import DPMSolverMultistepScheduler
import click
from models.models_dit3d_dropout import DiT3D_Diffuser
from modules.metrics import ChamferDistance, EMD
from modules.three_d_helpers import build_two_point_clouds, cylindrical_to_cartesian, angle_add
from datasets import dataset_mapper
from modules.class_mapping import class_mapping

def find_eligible_objects(dataloader, num_to_find=1, object_class='vehicle.car', min_points=None):
    targets = []

    for index, item in enumerate(dataloader):
        num_lidar_points = item['num_points'][0]
        item['index'] = index
        class_index = item['class'].item()

        if object_class != 'None' and class_mapping[object_class] != class_index:
            continue

        if num_lidar_points > min_points:
            targets.append(item)
        
        if len(targets) >= num_to_find:
            break

    return targets

def find_specific_objects(index, cfg):
    cfg['data']['data_dir'] = '/home/ekirby/scania/ekirby/datasets/bikes_from_nuscenes/bikes_from_nuscenes_train_val_reversed.json'
    module: LightningDataModule = dataset_mapper.dataloaders[cfg['data']['dataloader']](cfg)
    dataloader = module.train_dataloader(shuffle=False)
    objects = []
    for i, item in enumerate(dataloader):
        if i == index:
            item['index'] = index
            objects.append(item)
            break

    return objects

def visualize_step_t(x_t, pcd):
    points = x_t.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def calculate_bounds(wlh):
    half_wlh = wlh / 2
    min_bounds = -half_wlh
    max_bounds = half_wlh
    return min_bounds, max_bounds

def clip_points(points, min_bounds, max_bounds):
    clipped_points = torch.max(torch.min(points, max_bounds.unsqueeze(1)), min_bounds.unsqueeze(1)).squeeze(0)
    return clipped_points

def p_sample_loop(model: DiT3D_Diffuser, x_t, x_cond, x_class, batch_indices, viz_path=None):
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
    generate_viz = viz_path != None
    if generate_viz:
        viz_pcd = o3d.geometry.PointCloud()
        os.makedirs(f'{viz_path}/step_visualizations', exist_ok=True)

    for t in tqdm(range(len(model.dpm_scheduler.timesteps))):
        t = model.dpm_scheduler.timesteps[t].cuda()[None]

        with torch.no_grad():
            noise_t = model.classfree_forward(x_t, t, x_class, x_cond)
        input_noise = x_t

        x_t = model.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
    
        if generate_viz:
            viz = visualize_step_t(x_t.clone(), viz_pcd)
            print(f'Saving Visualization of Step {t}')
            o3d.io.write_point_cloud(f'{viz_path}/step_visualizations/object_gen_viz_step_{t[0]}.ply', viz)

    return x_t
        
def denoise_object_from_pcd(model: DiT3D_Diffuser, x_object, x_center, x_size, x_orientation, x_class, num_diff_samples, viz_path=None):
    torch.backends.cudnn.deterministic = True

    x_init = x_object.clone().cuda()    
    batch_indices = torch.zeros(x_init.shape[0]).long().cuda()

    if model.hparams['model']['cyclic_conditions'] > 0:
        if model.hparams['model']['relative_angles'] == True:
            x_cond = torch.cat((x_center, x_size),-1)
        else:
            x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
    else:
        x_cond = torch.hstack((x_center, x_size, x_orientation))

    x_cond = x_cond.cuda()
    x_class = x_class.cuda()

    local_chamfer = ChamferDistance()
    local_emd = EMD()
    x_gen_evals = []
    for i in tqdm(range(num_diff_samples)):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        x_feats = torch.randn(x_init.shape, device=model.device)
        x_gen_eval = p_sample_loop(model, x_feats, x_cond, x_class, batch_indices, viz_path=viz_path).squeeze(0).permute(1,0)
        x_gen_evals.append(x_gen_eval)    
        x_gen_pts = x_gen_eval[:, :3]
        object_pts = x_init.squeeze(0).permute(1,0)[:, :3]
        pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=x_gen_pts, object_pcd=object_pts)
        local_chamfer.update(pcd_gt, pcd_pred)
        local_emd.update(gt_pts=x_gen_pts, gen_pts=object_pts)

    best_index_cd = local_chamfer.best_index()
    best_index_emd = local_emd.best_index()
    print(f"Best seed cd: {best_index_cd}")
    print(f"Best seed emd: {best_index_emd}")
    x_gen_eval_cd = x_gen_evals[best_index_cd][:, :3].cpu().detach().numpy()
    x_gen_eval_emd = x_gen_evals[best_index_emd][:, :3].cpu().detach().numpy()
    
    cos_yaw = np.cos(x_orientation.item())
    sin_yaw = np.sin(x_orientation.item())
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    x_gen_eval_emd = np.dot(x_gen_eval_emd, rotation_matrix.T)

    og_pcd_cyl = x_center.clone()
    
    og_pcd_cyl[0,0] = angle_add(og_pcd_cyl[0,0], x_orientation.item())
    center =  cylindrical_to_cartesian(og_pcd_cyl)
    x_gen_eval_emd += center

    return x_gen_eval_cd, x_gen_eval_emd, x_object.cpu().detach().squeeze(0).permute(1,0).numpy()[:, :3]

def extract_object_info(object_info):
    x_object = object_info['pcd_object']
    x_center = object_info['center']
    x_size = object_info['size']
    x_orientation = object_info['orientation']
    x_class = object_info['class']

    return x_object, x_center, x_size, x_orientation, x_class

def find_pcd_and_test_on_object(dir_path, model, do_viz, objects, num_samples):
    for _, object_info in enumerate(objects):
        pcd, center_cyl, size, yaw, x_class = extract_object_info(object_info)
        x_gen_cd, x_gen_emd, x_orig = denoise_object_from_pcd(
            model,
            pcd, center_cyl, size, yaw, x_class,
            num_samples,
            viz_path=f'{dir_path}' if do_viz else None
        )

        cos_yaw = np.cos(yaw.item())
        sin_yaw = np.sin(yaw.item())
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        x_orig = np.dot(x_orig, rotation_matrix.T)
        og_pcd_cyl = center_cyl.clone()
        og_pcd_cyl[0,0] = angle_add(og_pcd_cyl[0,0], yaw.item())
        center = cylindrical_to_cartesian(og_pcd_cyl)
        x_orig += center

        np.savetxt(f'{dir_path}/{object_info["index"]}_generated_cd.txt', x_gen_cd)
        np.savetxt(f'{dir_path}/{object_info["index"]}_generated_emd.txt', x_gen_emd)
        np.savetxt(f'{dir_path}/{object_info["index"]}_original.txt', x_orig)

def find_pcd_and_interpolate_condition(dir_path, conditions, model, objects, do_viz):
    for object_info in objects:
        print(f'Generating using car info {object_info["index"]}')
        pcd, center_cyl, size, yaw, x_class = extract_object_info(object_info)
        
        og_pcd = pcd.cpu().detach().squeeze(0).permute(1,0).numpy()[:, :3]
        
        cos_yaw = np.cos(yaw.item())
        sin_yaw = np.sin(yaw.item())
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        og_pcd = np.dot(og_pcd, rotation_matrix.T)
        og_pcd_cyl = center_cyl.clone()
        og_pcd_cyl[0,0] += yaw.item()
        center = cylindrical_to_cartesian(og_pcd_cyl)
        og_pcd += center
        np.savetxt(f'{dir_path}/object_{object_info["index"]}_orig.txt', og_pcd)

        def do_gen(condition, index, center_cyl=center_cyl, size=size, yaw=yaw, pcd=pcd):
                _, x_gen, _ = denoise_object_from_pcd(
                    model=model,
                    x_object=pcd,
                    x_center=center_cyl,
                    x_size=size, 
                    x_orientation=yaw,
                    x_class=x_class,
                    num_diff_samples=1,
                    viz_path=f'{dir_path}' if do_viz else None
                )
                np.savetxt(f'{dir_path}/object_{object_info["index"]}_{condition}_interp_{index}.txt', x_gen)

        for condition in conditions:
            if condition == 'yaw':
                orig = yaw
                angles = np.linspace(start=orig, stop=orig*-1, num=5)
                print(f'Interpolating Yaw')
                for index, angle in enumerate(angles):
                    do_gen(condition, index=index, yaw=torch.from_numpy(angle))
            if condition == 'cylinder_angle':
                print(f'Interpolating Cylindrical Angle')
                start = center_cyl[:, 0]
                linspace_ring = np.linspace(start=start, stop=start + 2 * np.pi, num=5, endpoint=False)
                linspace_ring = (linspace_ring + np.pi) % (2 * np.pi) - np.pi
                for index, ring in enumerate(linspace_ring):
                    new_cyl = center_cyl.clone()
                    new_cyl[:,0] = ring.item()
                    do_gen(condition, index=index, center_cyl=new_cyl)
            if condition == 'cylinder_distance':
                print(f'Interpolating Cylindrical Distance')
                start = center_cyl[:, 1]
                linspace_dist = np.linspace(start=start, stop=start*4, num=4)
                for index, dist in enumerate(linspace_dist):
                    new_cyl = center_cyl.clone()
                    new_cyl[:,1] = dist.item()
                    orig_total_points = pcd.shape[-1]
                    subsampled_total_points = orig_total_points // (index + 2)
                    random_order = torch.randperm(orig_total_points)
                    new_pcd = pcd.clone()[:, :, random_order]
                    new_pcd = new_pcd[:, :, :subsampled_total_points]
                    do_gen(condition, index=index, center_cyl=new_cyl, pcd=new_pcd)
            
@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)'
            )
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt).',
              default=None
            )
@click.option('--output_path',
              '-o',
              type=str,
              help='path to save the generated point clouds',
              default='evaluations/generated_pcds'
            )
@click.option('--name',
              '-n',
              type=str,
              help='folder in which generated point clouds will be saved.',
              default=None
            )
@click.option('--task',
              '-t',
              type=str,
              help='Task to run. options are recreate or interpolate',
              default='recreate')
@click.option('--class_name',
              '-cls',
              type=str,
              help='Label of class to generate.',
              default='vehicle.car'
            )
@click.option('--split',
              '-s',
              type=str,
              help='Which split to take the conditioning information from.',
              default='train'
            )
@click.option('--min_points',
              '-m',
              type=int,
              help='Minimum number of points per cloud.',
              default=100
            )
@click.option('--do_viz',
              '-v',
              type=bool,
              help='Generate step visualizations (every step). True or False.',
              default=False
            )
@click.option('--examples_to_generate',
              '-e',
              type=int,
              help='Number of examples to generate',
              default=1
            )
@click.option('--num_samples',
              '-ds',
              type=int,
              help='Number of diffusion samples to take when recreating',
              default=10
            )
@click.option('--specific_obj_index',
              '-ind',
              type=int,
              help='Specific object index in sorted list to use',
              default=None,
            )
def main(config, weights, output_path, name, task, class_name, split, min_points, do_viz, examples_to_generate, num_samples, specific_obj_index):
    dir_path = f'{output_path}/{name}'
    os.makedirs(dir_path, exist_ok=True)
    cfg = yaml.safe_load(open(config))
    cfg['diff']['s_steps'] = 500
    cfg['train']['batch_size'] = 1
    model = DiT3D_Diffuser.load_from_checkpoint(weights, hparams=cfg).cuda()
    model.eval()

    if specific_obj_index != None:
        objects = find_specific_objects(specific_obj_index, cfg)
    else:
        module: LightningDataModule = dataset_mapper.dataloaders[cfg['data']['dataloader']](cfg)
        dataloader = module.train_dataloader() if split == 'train' else module.val_dataloader()
        objects = find_eligible_objects(dataloader, num_to_find=examples_to_generate, object_class=class_name, min_points=min_points)
    if task == 'recreate':
        find_pcd_and_test_on_object(dir_path=dir_path, model=model, objects=objects, do_viz=do_viz, num_samples=num_samples)
    if task == 'interpolate':
        find_pcd_and_interpolate_condition(
            dir_path=dir_path, 
            conditions=['cylinder_angle'], 
            model=model, 
            do_viz=do_viz, 
            objects=objects
        )

if __name__ == "__main__":
    main()

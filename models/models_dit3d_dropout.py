import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm
from os import makedirs, path
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import LightningDataModule
from diffusers import DPMSolverMultistepScheduler
from random import shuffle
from modules.scheduling import beta_func
from models.dit3d_pixart_pointnet_dropout import DiT3D_models_PixartDropoutCFG
from models.dit3d_cross_attn_dropout import DiT3D_models_CrossAttn
from models.dit3d_just_cross_attn import DiT3D_models_JustCrossAttn
from modules.metrics import ChamferDistance, EMD, RMSE
from modules.three_d_helpers import build_two_point_clouds
from copy import deepcopy
from modules.ema_utls import update_ema
import modules.class_mapping as class_mapping


class DiT3D_Diffuser(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        pretrained = self.hparams['model']['pretrained']
        attention_type = self.hparams['model']['attention']
        point_embeddings = self.hparams['model']['embeddings']
        model_size = self.hparams['model']['config']
        num_cyclic_conditions =  self.hparams['model']['cyclic_conditions']
        num_classes = 1 if self.hparams['train']['class_conditional'] == False else class_mapping.num_classes
        in_channels = 3 if 'in_channels' not in self.hparams['model'] else self.hparams['model']['in_channels']
        self.model = self.model_factory(pretrained, attention_type, point_embeddings, model_size, num_cyclic_conditions, num_classes, in_channels)

        self.chamfer_distance = ChamferDistance()
        self.emd = EMD()
        self.rmse = RMSE()

        self.w_uncond = self.hparams['train']['uncond_w']
        self.visualize = self.hparams['diff']['visualize']

        if self.hparams['train']['ema']:
            self.ema = deepcopy(self.model)
            for p in self.ema.parameters():
                p.requires_grad = False
            update_ema(self.ema, self.model, decay=0)
            self.ema.eval()
        else:
            self.ema = None

    def model_factory(self, pretrained, attention_type, point_embeddings, model_size, num_cyclic_conditions, num_classes, in_channels):
        factory = None
        if attention_type == 'justcross':
            factory = DiT3D_models_JustCrossAttn
        elif attention_type == 'cross' and point_embeddings == 'pointnet':
            factory = DiT3D_models_CrossAttn
        elif attention_type == 'pixart':
            factory = DiT3D_models_PixartDropoutCFG
        model = factory[model_size]
        return model(pretrained=pretrained, num_cyclic_conditions=num_cyclic_conditions, num_classes=num_classes, in_channels=in_channels)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None,None].cuda() * noise

    def classfree_forward(self, x_t, t, x_class, x_cond):
        x_c = self.forward(x_t, t, x_class, x_cond, force_dropout=False)            
        x_uc = self.forward(x_t, t,  x_class, x_cond, force_dropout=True)

        return x_uc + self.w_uncond * (x_c - x_uc)

    def visualize_step_t(self, x_t, gt_pts, pcd):
        points = x_t.detach().cpu().numpy()
        points = np.concatenate((points, gt_pts.detach().cpu().numpy()), axis=0)

        pcd.points = o3d.utility.Vector3dVector(points)
       
        colors = np.ones((len(points), 3))
        colors[:len(gt_pts)] = [1.,.3,.3]
        colors[len(gt_pts):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def p_sample_loop(self, x_t, x_class, x_cond, mask):
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = torch.ones(x_t.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            x_t *= mask
            noise_t = self.classfree_forward(x_t, t, x_class, x_cond)
            input_noise = x_t

            x_t = self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            
        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x, t, y, c, force_dropout=False):
        out = self.model(x, t, y, c, force_dropout)
        return out

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        x_object = batch['pcd_object'].cuda()
        padding_mask = batch['padding_mask'][:, None, :]
        noise = torch.randn(x_object.shape, device=self.device) * padding_mask
        
        # sample step t
        t = torch.randint(0, self.t_steps, size=(noise.shape[0],))
        # sample q at step t
        t_sample = self.q_sample(x_object, t, noise).float() * padding_mask
        t = t.cuda()

        x_center = batch['center']
        x_size = batch['size']
        x_orientation = batch['orientation']

        x_class = batch['class']

        if self.hparams['model']['cyclic_conditions'] > 0:
            if self.hparams['model']['relative_angles'] == True:
                x_cond = torch.cat((x_center, x_size),-1)
            else:
                x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
        else:
            x_cond = torch.hstack((x_center, x_size, x_orientation))

        denoise_t = self.forward(t_sample, t, x_class, x_cond) * padding_mask
        loss_mse = self.p_losses(denoise_t, noise)
        loss_mean = (denoise_t.mean())**2
        loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_noise.var())
        self.log('train/std', std_noise.std())

        return loss
    
    def on_after_backward(self) -> None:
        if self.ema != None:
            update_ema(self.ema, self.model)

    def validation_step(self, batch:dict, batch_idx):
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()
        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']

            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']

            if self.hparams['model']['cyclic_conditions'] > 0:
                if self.hparams['model']['relative_angles'] == True:
                    x_cond = torch.cat((x_center, x_size),-1)
                else:
                    x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))
            
            padding_mask = batch['padding_mask']
            x_t = torch.randn(x_object.shape, device=self.device)
            x_gen_eval = self.p_sample_loop(x_t, batch['class'], x_cond, padding_mask[:, None, :]).permute(0,2,1).squeeze(0)
            x_object = x_object.permute(0,2,1).squeeze(0)
            
            for pcd_index in range(batch['num_points'].shape[0]):
                mask = padding_mask[pcd_index].int() == True
                object_pcd = x_object[pcd_index].squeeze(0)[mask]
                genrtd_pcd = x_gen_eval[pcd_index].squeeze(0)[mask]
                
                object_points = object_pcd[:, :3]
                genrtd_points = genrtd_pcd[:, :3]
                
                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_points, object_pcd=object_points)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.emd.update(object_points, genrtd_points)
                if genrtd_points.shape[1] == 4:
                    object_intensity = object_pcd[:, 3]
                    genrtd_intensity = genrtd_pcd[:, 3]
                    self.rmse.update(object_intensity, genrtd_intensity)

        cd_mean, cd_std = self.chamfer_distance.compute()
        emd_mean, emd_std = self.emd.compute()
        rmse_mean, rmse_std = self.rmse.compute()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/emd_mean', emd_mean, on_step=True)
        self.log('val/emd_std', emd_std, on_step=True)
        self.log('val/intensity_mean', rmse_mean)
        self.log('val/intensity_std', rmse_std)
        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/emd_mean': emd_mean, 'val/emd_std': emd_std, 'val/intensity_mean':rmse_mean, 'val/intensity_std':rmse_std}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']

            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']

            if self.hparams['model']['cyclic_conditions'] > 0:
                if self.hparams['model']['relative_angles'] == True:
                    x_cond = torch.cat((x_center, x_size),-1)
                else:
                    x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))

            padding_mask = batch['padding_mask']

            x_gen_evals = []
            num_val_samples = self.hparams['diff']['num_val_samples']
            for _ in range(num_val_samples):
                self.dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=self.t_steps,
                    beta_start=self.hparams['diff']['beta_start'],
                    beta_end=self.hparams['diff']['beta_end'],
                    beta_schedule='linear',
                    algorithm_type='sde-dpmsolver++',
                    solver_order=2,
                )
                self.dpm_scheduler.set_timesteps(self.s_steps)
                self.scheduler_to_cuda()
                x_t = torch.randn(x_object.shape, device=self.device)
                x_gen_eval = self.p_sample_loop(x_t, batch['class'], x_cond, padding_mask[:, None, :]).permute(0,2,1)
                x_gen_evals.append(x_gen_eval)

            x_object = x_object.permute(0,2,1)
            for pcd_index in range(batch['num_points'].shape[0]):
                mask = padding_mask[pcd_index].int() == True
                object_pcd = x_object[pcd_index].squeeze(0)[mask]
                object_points = object_pcd[:, :3]
                
                local_chamfer = ChamferDistance()
                local_emd = EMD()
                for generated_pcds in x_gen_evals:
                    genrtd_pcd = generated_pcds[pcd_index].squeeze(0)[mask]
                    genrtd_points = genrtd_pcd[:, :3]
                    
                    pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_points, object_pcd=object_points)

                    local_chamfer.update(pcd_gt, pcd_pred)
                    local_emd.update(gt_pts=object_points, gen_pts=genrtd_points)

                best_index_cd = local_chamfer.best_index()
                genrtd_pcd_cd = x_gen_evals[best_index_cd][pcd_index].squeeze(0)[mask][:, :3]
                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_pcd_cd, object_pcd=object_points)
                self.chamfer_distance.update(pcd_gt, pcd_pred)

                best_index_emd = local_emd.best_index()
                genrtd_pcd_emd = x_gen_evals[best_index_emd][pcd_index].squeeze(0)[mask][:, :3]
                self.emd.update(object_points, genrtd_pcd_emd)
                
                if genrtd_pcd_emd.shape[1] == 4:
                    genrtd_intensity = genrtd_pcd_emd[:, 3]
                    object_intensity = object_pcd[:, 3]
                    self.rmse.update(object_intensity, genrtd_intensity)

        cd_mean, cd_std = self.chamfer_distance.compute()
        emd_mean, emd_std = self.emd.compute()
        rmse_mean, rmse_std = self.rmse.compute()

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/emd_mean', emd_mean, on_step=True)
        self.log('test/emd_std', emd_std, on_step=True)
        self.log('test/intensity_mean', rmse_mean, on_step=True)
        self.log('test/intensity_std', rmse_std, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/emd_mean': emd_mean, 'test/emd_std': emd_std, 'test/intensity_mean':rmse_mean, 'test/intensity_std':rmse_std,}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = {
            # 'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer]

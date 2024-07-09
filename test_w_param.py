import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
from datasets.dataset_mapper import dataloaders
from models.models_dit3d import DiT3D_Diffuser

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to checkpoint file (.ckpt) to load weights.',
              default=None)
def main(config, weights):
    cfg = yaml.safe_load(open(config))
    og_id = cfg['experiment']['id']
    w_params = np.linspace(start=0.0, stop=2.0, num=3)
    for w in w_params:
        cfg['train']['uncond_w'] = w
        cfg['experiment']['id'] = og_id + f'_w_{cfg["train"]["uncond_w"]}'
        print("TESTING ", cfg['experiment']['id'])
        model = DiT3D_Diffuser.load_from_checkpoint(weights, hparams=cfg)
        dl = cfg['data']['dataloader']
        data = dataloaders[dl](cfg)

        #Add callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_saver = ModelCheckpoint(
                                    dirpath='checkpoints/'+cfg['experiment']['id'],
                                    filename=cfg['experiment']['id']+'_{epoch:02d}',
                                    save_last=True,
                                )

        tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                                default_hp_metric=False)
        #Setup trainer
        if torch.cuda.device_count() > 1:
            cfg['train']['n_gpus'] = torch.cuda.device_count()
            trainer = Trainer(
                            devices=cfg['train']['n_gpus'],
                            logger=tb_logger,
                            log_every_n_steps=100,
                            max_epochs= cfg['train']['max_epoch'],
                            callbacks=[lr_monitor, checkpoint_saver],
                            check_val_every_n_epoch=10,
                            num_sanity_val_steps=0,
                            limit_val_batches=1,
                            accelerator='gpu',
                            strategy="ddp",
                            )
        else:
            trainer = Trainer(
                            accelerator='gpu',
                            devices=cfg['train']['n_gpus'],
                            logger=tb_logger,
                            log_every_n_steps=100,
                            max_epochs= cfg['train']['max_epoch'],
                            callbacks=[lr_monitor, checkpoint_saver],
                            check_val_every_n_epoch=10,
                            num_sanity_val_steps=0,
                            limit_val_batches=1,
                    )
        trainer.test(model, data)

if __name__ == "__main__":
    main()

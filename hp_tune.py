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

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--hp_type',
              '-h1',
              type=str,
              help='type of hyper-parameter (train, diff, model, etc.).',
              default='train'
              )
@click.option('--hp_name',
              '-h2',
              type=str,
              help='name of hyper-parameter.',
              default='lr'
              )
@click.option('--value',
              '-v',
              type=float,
              help='value of hyper parameter.',
              default=None
              )
def main(config, checkpoint, hp_type, hp_name, value):
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    cfg[hp_type][hp_name] = value
    cfg['experiment']['id'] += f'_{hp_name}_{value}'
    model = DiT3D_Diffuser(cfg)
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
                        limit_test_batches=10,
                        limit_val_batches=1,
                )



    trainer.fit(model, data, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()

import click
from os.path import join, dirname, abspath
from os import environ, makedirs
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
from datasets.dataset_mapper import dataloaders
from models.models_dit3d import DiT3D_Diffuser
from models.models_dit3d_dropout import DiT3D_Diffuser as DiT3D_Diffuser_dropout

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
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
@click.option('--test_with_ema', '-te', is_flag=True, help='test with ema')
def main(config, weights, checkpoint, test, test_with_ema):
    if not test:
        set_deterministic()

    cfg = yaml.safe_load(open(config))

    #Load data and model
    if weights is None:
        model = DiT3D_Diffuser(cfg) if 'improved_cfg' not in cfg['model'] else DiT3D_Diffuser_dropout(cfg)
    else:
        if test:
            ckpt_cfg = yaml.safe_load(open(config))
            cfg = ckpt_cfg

        model = DiT3D_Diffuser if 'improved_cfg' not in cfg['model'] else DiT3D_Diffuser_dropout
        model = model.load_from_checkpoint(weights, hparams=cfg)
        if test_with_ema:
            model.model = model.ema

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
                        num_sanity_val_steps=2,
                        limit_val_batches=2,
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
                        num_sanity_val_steps=2,
                        limit_val_batches=2,
                )


    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data, ckpt_path=checkpoint)

if __name__ == "__main__":
    main()

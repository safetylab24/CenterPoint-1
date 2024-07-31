import faulthandler
import torch
from pathlib import Path
from lightning import Trainer
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import os
from det3d.torchie import Config
import torch
from det3d.models.detectors.modelmodule import CPModel
from det3d.models.detectors.datamodule import DataModule
from det3d.datasets import build_dataset
from torch.utils.data import DataLoader
from det3d.models import build_detector
from det3d.torchie.parallel import collate_kitti

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    cfg = Config.fromfile(
        '/home/vxm240030/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py')

    faulthandler.enable()
    torch.cuda.empty_cache()

    logger = TensorBoardLogger(
        save_dir=cfg.work_dir,
        name='train',
        default_hp_metric=False,
    )

    hyperparameters = {
        'epochs': cfg.total_epochs,
        'batch_size': cfg.data.samples_per_gpu,
        'lr_max': cfg.lr_config.lr_max,
        'base_momentum': cfg.lr_config.moms[1],
        'max_momentum': cfg.lr_config[0],
        'weight_decay': cfg.optimizer.wd,
        'num_workers': cfg.data.samples_per_gpu,
    }

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    modelmodule = CPModel(model, cfg)
    datamodule = DataModule(cfg)

    logger.log_hyperparams(hyperparameters)

    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
        log_weight_decay=True)

    checkpointer = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / 'checkpoints',
        filename='{epoch}-{step}',
        verbose=True,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=True,
        every_n_epochs=1,
    )

    model_summary = ModelSummary(max_depth=4)

    trainer = Trainer(
        accelerator='gpu',
        devices=[0, 1, 2, 3],
        max_epochs=cfg.total_epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpointer, lr_monitor, model_summary],
    )

    # print('Hyperparameters:', hyperparameters)
    # print('Logger version:', logger.version)
    # print('Log dir:', logger.log_dir)

    trainer.fit(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=cfg.load_from
    )

    print('TRAINING DONE')
    print('\n=========================')


if __name__ == '__main__':
    main()

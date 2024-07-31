from lightning import LightningDataModule
from det3d.datasets import build_dataset
from det3d.torchie.parallel import collate_kitti
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.data_train = build_dataset(self.cfg.data.train)
            self.data_val = build_dataset(self.cfg.data.val)
    
    def train_dataloader(self):
        data_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.data.samples_per_gpu,
            shuffle=True,
            num_workers=self.cfg.data.workers_per_gpu,
            collate_fn=collate_kitti,
            drop_last=True,
            pin_memory=True,
        )
        return data_loader
    
    def val_dataloader(self):
        data_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.data.samples_per_gpu,
            shuffle=False,
            num_workers=self.cfg.data.workers_per_gpu,
            collate_fn=collate_kitti,
            drop_last=True,
            pin_memory=True,
        )
        return data_loader

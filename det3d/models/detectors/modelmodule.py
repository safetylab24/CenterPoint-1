import torch
from lightning import LightningModule
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from math import sqrt
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from nuscenes import NuScenes
import AveragePrecision


class CPModel(LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.average_precision = AveragePrecision()
        self.preds = []

    def training_step(self, batch, batch_idx):
        loss, pred_boxes = self.model(batch)
        loss, loss_elems = self.parse_second_losses(loss)

        # compute ap
        ap = self.average_precision(
            pred_boxes[0]['box3d_lidar'], pred_boxes[0]['scores'], batch['gt_boxes_and_cls'][..., :-1].squeeze(0))

        self.log('train_ap', ap)

        self.log("train_loss", loss)
        self.log("train_loss_loc", loss_elems['loc_loss'])
        self.log("train_loss_hm", loss_elems['hm_loss'])

        pred_boxes = pred_boxes[0]['box3d_lidar']
        label_boxes = batch['gt_boxes_and_cls'][..., :-1].squeeze(0)
        # remove any rows with all zeros
        label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        ap = self.average_precision(pred_boxes, label_boxes)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_boxes = self.model(batch)
        loss, loss_elems = self.parse_second_losses(loss)

        # compute ap
        ap = self.average_precision(
            pred_boxes[0]['box3d_lidar'], pred_boxes[0]['scores'], batch['gt_boxes_and_cls'][..., :-1].squeeze(0))

        self.log('val_ap', ap)

        self.log("val_loss", loss)
        self.log("val_loss_loc", loss_elems['loc_loss'])
        self.log("val_loss_hm", loss_elems['hm_loss'])

        pred_boxes = pred_boxes[0]['box3d_lidar']
        label_boxes = batch['gt_boxes_and_cls'][..., :-1].squeeze(0)
        # remove any rows with all zeros
        label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        ap = self.average_precision(pred_boxes, label_boxes)

        return loss

    def test_step(self, batch, batch_idx):
        preds = self.model(batch, return_loss=False)
        preds_boxes = preds[0]['box3d_lidar']
        # TODO: display confidence score in visualization next to corresponding box
        # label_boxes = batch['gt_boxes_and_cls'][..., :-1].squeeze(0)
        # remove any rows with all zeros
        # label_boxes = label_boxes[~torch.all(label_boxes == 0.0, dim=1)]
        # self.visualize(preds_boxes, label_boxes, batch['metadata'][0]['token'])

        # return boxes

        for i in range(preds_boxes.shape[0]):
            attr = 'vehicle.moving' if sqrt(
                preds_boxes[i, 6] ** 2 + preds_boxes[i, 7] ** 2) > 0.2 else 'vehicle.parked'
            pred = {
                'sample_token': batch['metadata'][0]['token'],
                'translation': preds_boxes[i, :3].tolist(),
                'size': preds_boxes[i, 3:6].tolist(),
                'rotation': R.from_euler('z', preds_boxes[i, -1].item(), degrees=False).as_quat().tolist(),
                'velocity': preds_boxes[i, 6:8].tolist(),
                'detection_name': 'car',
                'detection_score': preds[0]['scores'][i].item(),
                'attribute_name': attr
            }
            self.preds.append(pred)

    def on_test_epoch_end(self):
        with open(os.path.join(self.logger.log_dir, "prediction.pkl"), "wb") as f:
            pickle.dump(self.preds, f)

        nusc = NuScenes(version='v1.0-trainval',
                        dataroot='/home/vxm240030/nuscenes', verbose=True)

        eval = DetectionEval(nusc, config_factory("cvpr_2019"), os.path.join(
            self.logger.log_dir, "prediction.pkl"), 'val', self.logger.log_dir, verbose=True)

        eval.main()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            amsgrad=self.cfg.optimizer.amsgrad,
            weight_decay=self.cfg.optimizer.wd,
            betas=(0.9, 0.99),
            lr=self.cfg.lr_scheduler.lr_max)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr_scheduler.lr_max,
            total_steps=self.trainer.estimated_stepping_batches,
            max_momentum=self.cfg.lr_scheduler.moms[0],
            base_momentum=self.cfg.lr_scheduler.moms[1],
            div_factor=self.cfg.lr_scheduler.div_factor,
            pct_start=self.cfg.lr_scheduler.pct_start)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'frequency': 1,
                'interval': 'step'},
        }

    def parse_second_losses(self, losses):
        log_vars = OrderedDict()
        loss = sum(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == 'loc_loss_elem':
                log_vars[loss_name] = [[i.item() for i in j]
                                       for j in loss_value]
            else:
                log_vars[loss_name] = [i.item() for i in loss_value]

        return loss, log_vars

    def visualize(self, pred_boxes_3d, label_boxes_3d, token):
        pred_boxes_2d = []
        label_boxes_2d = []
        for box in pred_boxes_3d:
            x, y, z, w, l, h, vel_x, vel_y, yaw = box
            pred_boxes_2d.append((x, y, w, l))
        for box in label_boxes_3d:
            x, y, z, w, l, h, vel_x, vel_y, yaw = box
            label_boxes_2d.append((x, y, w, l))
        # Create a plot
        fig, ax = plt.subplots()

        # Plot each pred bounding box
        for bbox in pred_boxes_2d:
            c_x, c_y, w, l = bbox
            lower_left_x = c_x - w / 2
            lower_left_y = c_y - l / 2
            w = w.item()
            l = l.item()
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), w, l, yaw, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot each label bounding box
        for bbox in label_boxes_2d:
            c_x, c_y, w, l = bbox
            lower_left_x = c_x - w / 2
            lower_left_y = c_y - l / 2
            w = w.item()
            l = l.item()
            rect = patches.Rectangle((lower_left_x.item(), lower_left_y.item(
            )), w, l, yaw, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Plot the ego vehicle at the origin
        ax.plot(0, 0, 'bo')  # blue dot

        # Set plot limits
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        # Set labels and title
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('2D Bounding Boxes Relative to Ego Vehicle')

        # Save the plot as an image
        path = Path(self.logger.log_dir) / 'bounding_boxes_plots'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path / f'bounding_boxes_plot_{token}.png')
        plt.close()

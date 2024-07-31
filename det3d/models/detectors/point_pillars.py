from torch import nn
from torchmetrics import Metric
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy
import torch
from .decoder import Decoder
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchmetrics import Metric
import torch
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # self.f1 = F1Score3D()
        self.mlp = MLP(num_hidden=4)
        self.decoder = Decoder(dim=128, blocks=[128, 128, 64])
        self.decoder.load_state_dict(torch.load('./decoder.pth'))
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=384, kernel_size=(
            768, 3, 3), stride=(768, 1, 1), padding=(0, 1, 1))

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        ctp_bev = x

        # fusion
        cvt_bev = torch.tensor([]).to(x.device)
        for batch in example['metadata']:
            bev_loaded_batch = torch.load(
                f"/home/vxm240030/CenterPoint/predictions/{batch['token']}.pth").to(x.device)
            cvt_bev = torch.cat((cvt_bev, bev_loaded_batch), dim=0)
        cvt_bev = self.decoder(cvt_bev)

        # interpolate cvt_bev to match sizes
        cvt_bev = self.mlp(cvt_bev)
        cvt_bev = F.interpolate(cvt_bev, size=(128, 128))

        # fuse
        bev_fused = torch.cat((cvt_bev, ctp_bev), dim=1).contiguous()
        bev_fused = bev_fused.unsqueeze(1)  # Bx1xDxHxW

        # Apply the convolution
        bev_fused = self.conv3d(bev_fused)  # Bx128x1xHxW

        # Remove the singleton dimension
        # Now the shape is (4, 128, 128, 128)
        bev_fused = bev_fused.squeeze(2)

        preds, _ = self.bbox_head(bev_fused)
        
        loss = self.bbox_head.loss(example, preds, self.test_cfg)
        boxes = self.bbox_head.predict(example, preds, self.test_cfg)
        return loss, boxes

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x
        # ctp_bev = x

        # # fusion
        # cvt_bev = torch.tensor([]).to(x.device)
        # for batch in example['metadata']:
        #     bev_loaded_batch = torch.load(
        #         f"/home/vxm240030/CenterPoint/predictions/{batch['token']}.pth").to(x.device)
        #     cvt_bev = torch.cat((cvt_bev, bev_loaded_batch), dim=0)
        # cvt_bev = self.decoder(cvt_bev)

        # # interpolate cvt_bev to match sizes
        # cvt_bev = self.mlp(cvt_bev)
        # cvt_bev = F.interpolate(cvt_bev, size=(360, 360))

        # # fuse
        # bev_fused = torch.cat((cvt_bev, ctp_bev), dim=1).contiguous()
        # bev_fused = bev_fused.unsqueeze(1) # Bx1xDxHxW

        # # Apply the convolution
        # bev_fused = self.conv3d(bev_fused) # Bx128x1xHxW

        # # Remove the singleton dimension
        # # Now the shape is (4, 128, 128, 128)
        # bev_fused = bev_fused.squeeze(2)

        # preds, _ = self.bbox_head(bev_fused)
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {}
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        loss = self.bbox_head.loss(example, new_preds, self.test_cfg)
        # self.f1_metric(boxes, example['gt_boxes_and_cls'])
        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, loss
        else:
            return boxes, bev_feature, None


class MLP(nn.Module):
    def __init__(self, num_hidden):
        from torchvision.ops import MLP
        super().__init__()
        self.out_dim = 384
        self.mlp = MLP(in_channels=64, hidden_channels=[
                       self.out_dim for _ in range(num_hidden)])

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        # Flatten the spatial dimensions: BxCxHW
        x = x.view(B, C, H * W).permute(0, 2, 1)  # BxHWxC
        x = self.mlp(x)  # Apply MLP: Bx625x128
        x = x.permute(0, 2, 1).view(B, self.out_dim, H, W)  # BxC'xHXW
        # Upsample spatial dimensions to 128x128
        return x

import torch
from torchmetrics import Metric
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class AveragePrecision(Metric):
    def __init__(self, iou_threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.add_state("pred_boxes", default=[], dist_reduce_fx="cat")
        self.add_state("pred_scores", default=[], dist_reduce_fx="cat")
        self.add_state("gt_boxes", default=[], dist_reduce_fx="cat")

    def update(self, pred_boxes, pred_scores, gt_boxes):
        self.pred_boxes.append(pred_boxes)
        self.pred_scores.append(pred_scores)
        self.gt_boxes.append(gt_boxes)

    def compute(self):
        pred_boxes = torch.cat(self.pred_boxes, dim=0)
        pred_scores = torch.cat(self.pred_scores, dim=0)
        gt_boxes = torch.cat(self.gt_boxes, dim=0)
        
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]

        tp = torch.zeros(len(pred_boxes), dtype=torch.float32, device=pred_boxes.device)
        fp = torch.zeros(len(pred_boxes), dtype=torch.float32, device=pred_boxes.device)
        matched_gts = set()

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gts:
                tp[i] = 1
                matched_gts.add(best_gt_idx)
            else:
                fp[i] = 1

        cumulative_tp = torch.cumsum(tp, dim=0)
        cumulative_fp = torch.cumsum(fp, dim=0)

        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / len(gt_boxes)

        # Adding (0,1) point to the precision-recall curve
        precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])
        recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])

        # AP is the area under the precision-recall curve
        ap = torch.trapz(precision, recall)
        
        return ap

    @staticmethod
    def compute_iou(box1, box2):
        iou_3d = boxes_iou3d_gpu(box1, box2)
        return iou_3d

# Example usage:
# preds = (pred_boxes, pred_scores)
# targets = gt_boxes

# metric = AveragePrecision()
# metric.update(preds, targets)
# ap = metric.compute()
# print(f"Average Precision: {ap}")
# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator, DetectionValidatorCustom
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, OBBMetricsCustom, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images
import numpy as np


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml")
        validator = OBBValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        #print('class OBBValidator: __init__() called')
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)
        self.printed = False

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        #print('class OBBValidator: init_metrics() called')
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        if not self.printed:
            #print('class OBBValidator: postprocess() called')
            self.printed = True

        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=True,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Perform computation of the correct prediction matrix for a batch of detections and ground truth bounding boxes.

        Args:
            detections (torch.Tensor): A tensor of shape (N, 7) representing the detected bounding boxes and associated
                data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
            gt_bboxes (torch.Tensor): A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
                represented as (x1, y1, x2, y2, angle).
            gt_cls (torch.Tensor): A tensor of shape (M,) representing class labels for the ground truth bounding boxes.

        Returns:
            (torch.Tensor): The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
                Union) levels for each detection, indicating the accuracy of predictions compared to the ground truth.

        Example:
            ```python
            detections = torch.rand(100, 7)  # 100 sample detections
            gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
            gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
            correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
            ```

        Note:
            This method relies on `batch_probiou` to calculate IoU between detections and ground truth bounding boxes.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
        if not self.printed:
            print('class OBBValidator: _prepare_pred() called')
            self.printed = True
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        import numpy as np

        from ultralytics.engine.results import Results

        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        # xywh, r, conf, cls
        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=obb,
        ).save_txt(file, save_conf=save_conf)

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
    

class OBBValidatorCustom(DetectionValidatorCustom):
    """
    A custom class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidatorCustom

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml")
        validator = OBBValidatorCustom(args=args)
        validator(model=args["model"])
        ```
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        #print('class OBBValidatorCustom: __init__() called')
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetricsCustom(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)
        self.printed = False # DEBUG variable
        self.target_difficulty = None # here we can change the eval difficulty

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        #print('class OBBValidatorCustom: init_metrics() called')
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        if not self.printed:
            #print('class OBBValidatorCustom: postprocess() called')
            self.printed = True
        arg = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=True,
        )
        return arg 

    def _process_batch(self, detections, gt_bboxes, gt_cls, difficulty):
        """Process batch with difficulty filtering."""
        # 1. Calculate IoU with all ground truth boxes first
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        # 2. Get initial matches with all ground truth
        return self.match_predictions(detections[:, 5], gt_cls, iou)
        

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        # 12.03.25 extract the difficulty from batch dictionary 
        difficulty = batch["difficulty"][si] # si = might be sample index?

        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
    
        # DEBUG
        #print("Difficulty values:", difficulty) 
        #print("Difficulty length:", len(difficulty))

        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        return {"cls": cls, 
                "bbox": bbox, 
                "ori_shape": ori_shape, 
                "imgsz": imgsz, 
                "ratio_pad": ratio_pad, 
                "difficulty": difficulty # 12.03.25 add difficulty to dict
        }

    def get_kitti_obj_level(self, difficulty):
        """
        Determines the difficulty level of objects based on the KITTI dataset's criteria.

        This function processes a given `difficulty` array containing metadata about objects in a scene.
        Each object's difficulty level is calculated based on its bbox, truncation, and occlusion values.
        The function assigns one of the following difficulty levels for each object:
            - `1`: Easy
            - `2`: Moderate
            - `3`: Hard
            - `0`: Unknown

        Parameters:
            difficulty (np.ndarray): A 2D array where each row corresponds to an object and contains:
                - Column 0-3: bbox (float)
                - Column 4: truncation (float)
                - Column 5: occlusion (int)

        Returns:
            List[int]: A list of difficulty levels for all objects in the input array.
                    Each value in the list corresponds to the calculated difficulty level of an object.
        """
        difficulty_levels = []

        if isinstance(difficulty, np.ndarray) and difficulty.shape[1] > 0:
            for obj_idx in range(difficulty.shape[0]):
                # Check format based on number of columns
                if difficulty.shape[1] >= 6:  # New format with full bbox
                    y1 = difficulty[obj_idx, 1]  # bbox_y1
                    y2 = difficulty[obj_idx, 3] # bbox_y2
                    height = float(y2) - float(y1) + 1
                    truncation = difficulty[obj_idx, 4]  # truncation
                    occlusion = difficulty[obj_idx, 5]  # occlusion
                elif difficulty.shape[1] >= 3:  # Old format or other format with at least height, trunc, occl
                    height = difficulty[obj_idx, 0]  # height is first column
                    truncation = difficulty[obj_idx, 1]  # truncation
                    occlusion = difficulty[obj_idx, 2]  # occlusion
                # Official KITTI difficulties:
                if height >= 40 and truncation <= 0.15 and occlusion <= 0: # fully visible
                    level = 1 # Easy
                elif height >=25 and truncation <=0.3 and occlusion <= 1: # Partly occluded
                    level = 2 # Moderate
                elif height >= 25 and truncation <= 0.5 and occlusion <= 2: # Largely occluded
                    level = 3 # Hard
                else:
                    level = 0 # Unknown
                
                difficulty_levels.append(level)

        return difficulty_levels

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        import numpy as np

        from ultralytics.engine.results import Results

        # xywh, r, conf, cls
        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)

        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=obb,
        ).save_txt(file, save_conf=save_conf)

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats

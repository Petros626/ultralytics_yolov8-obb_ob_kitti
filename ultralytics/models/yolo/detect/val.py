# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator, BaseValidatorCustom
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images, plot_matches
from ultralytics.utils.metrics import batch_probiou


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolo11n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        #print('class DetectionValidator: __init__() called')
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95, 0.05 steps
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)] if self.args.save_hybrid else []

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        #print('class DetectionValidator: init_metrics() called')
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        if self.args.plots and self.args.visualize:
            (self.save_dir / "visualizations").mkdir(exist_ok=True)
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc, conf=self.args.conf, save_matches=self.args.plots and self.args.visualize
        )
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        #print('class DetectionValidator: get_desc() called')
        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP70", "mAP50-95)") # 25.01.25 Updated by mAP70
    

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics.""" 
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bbox, gt_cls=cls, im_name=Path(batch["im_file"][si]).name
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls, im_name=Path(batch["im_file"][si]).name)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.plots and self.args.visualize:
                plot_matches(self, batch, preds, si)
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        #print('class DetectionValidator: finalize_metrics() called')
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    # 25.01.25 Updated by mAP70
    def print_results(self):
        """Prints training/validation set metrics per class."""
        #print('class DetectionValidator: print_results() called')
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format, (p, r, map50, map70, map50-95)
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        #print('class DetectionValidator: build_dataset() called')
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        #print('class DetectionValidator: get_dataloader() called')
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats


class DetectionValidatorCustom(BaseValidatorCustom):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        #print('class DetectionValidatorCustom: __init__() called')
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None # no. targets per class
        self.nt_per_image = None # no. targets per image
        self.nd_per_class = None # no. difficulties per class
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95, 0.05 steps
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )
        
    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid and self.args.task == "detect":
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)] if self.args.save_hybrid else []

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        #print('class DetectionValidatorCustom: init_metrics() called')
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        if self.args.plots and self.args.visualize:
            (self.save_dir / "visualizations").mkdir(exist_ok=True)
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc, conf=self.args.conf, save_matches=self.args.plots and self.args.visualize
        )
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[], difficulty=[]) # add list for difficulty
        self.objects_per_class_and_difficulty = {} # 15.03.25 difficulty statistic dict
    
    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        #print('class DetectionValidatorCustom: get_desc() called')
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "AP50", "AP70)") # custom
        
    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def get_kitti_obj_level(self, si, difficulty):
        """Get the object level from the difficulty metadata in data."""
        return difficulty

    def update_metrics(self, preds, batch): # maybe here the difficulty can be calculated
            """
            Update metrics with new predictions and ground truth.

            Args:
                preds (List[torch.Tensor]): List of predictions from the model.
                batch (dict): Batch data containing ground truth.
            """
            for si, pred in enumerate(preds):
                self.seen += 1 # no. all validation images
                npr = len(pred) # number predictions
                stat = dict(
                    conf=torch.zeros(0, device=self.device),
                    pred_cls=torch.zeros(0, device=self.device),
                    tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                )

                pbatch = self._prepare_batch(si, batch) # added difficulty key from batch, see _prepare_batch() in obb/val.py
                cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
                # 12.03.25 Retrieve difficulty value from pbatch dictionary. Calculate difficulty levels for each object in the current sample
                difficulty = pbatch.pop("difficulty")
                difficulty_levels = torch.tensor(self.get_kitti_obj_level(difficulty), 
                                       device=self.device)

                # DEBUG
                #print("Difficulty_levels values:", difficulty_levels) 
                #print("Difficulty_levels length:", len(difficulty_levels))            

                nl = len(cls)  # number filtered labels
                stat["target_cls"] = cls
                stat["target_img"] = cls.unique()
                stat["difficulty"] = difficulty_levels # maybe can be deleted

                # Update global metrics
                if npr == 0:
                    if nl:
                        for k in self.stats.keys():
                            self.stats[k].append(stat[k]) # tp, conf, pred_cls, target_cls, target_img, difficulty
                        if self.args.plots:
                            self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bbox, gt_cls=cls, im_name=Path(batch["im_file"][si]).name
                        )
                            #self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                    continue

                # Predictions
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = self._prepare_pred(pred, pbatch) # in class OBBValidatorCustom
                stat["conf"] = predn[:, 4] # confidence current sample
                stat["pred_cls"] = predn[:, 5] # class current sample

                # Evaluate
                if nl: # number labels
                    stat["tp"] = self._process_batch(predn, bbox, cls, difficulty)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls, im_name=Path(batch["im_file"][si]).name)
                    #self.confusion_matrix.process_batch(predn, bbox, cls)
                for k in self.stats.keys(): # 'conf', 'pred_cls', 'tp', 'target_cls', 'target_img', 'difficulty'
                    self.stats[k].append(stat[k])
                    
                    #current_value = self.stats[k][-1]
                    #print(f"Key: {k}, Shape: {current_value.shape}")
                    #print(f"Key: {k}, Content: {len(self.stats[k])}")

                # Save
                if self.args.plots and self.args.visualize:
                    plot_matches(self, batch, preds, si)
                if self.args.save_json:
                    self.pred_to_json(predn, batch["im_file"][si])
                if self.args.save_txt:
                    self.save_one_txt(
                        predn,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                    )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        #print('class DetectionValidatorCustom: finalize_metrics() called')
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        #print('class DetectionValidatorCustom: get_stats() called')
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy

        # DEBUG
        #difficulty_levels = stats["difficulty"] # 16907 objects in 3699 val. data
        #unique_difficulties, counts = np.unique(stats["difficulty"], return_counts=True)
        #print(f"Unique difficulties: {(unique_difficulties)}")
        #print(f"Counts per difficulty: {(counts)}")

        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc) # no. targets per class
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc) # no. targets per image
        self.nd_per_class = np.bincount(stats["difficulty"].astype(int), minlength=4) # no. difficulties per class
        
        # 15.03.25 processes statistics for different classes and their difficulty levels
        for class_idx in range(self.nc):
            if class_idx == 0 and 0 in self.names:
                continue
            class_mask = stats["target_cls"] == class_idx
            class_difficulties = stats["difficulty"][class_mask]
            difficulty_counts = np.bincount(class_difficulties.astype(int), minlength=4)
            self.objects_per_class_and_difficulty[self.names[class_idx]] = {
                "Easy": difficulty_counts[1],
                "Moderate": difficulty_counts[2],
                "Hard": difficulty_counts[3],
                "Unknown": difficulty_counts[0],
            }

        stats.pop("target_img", None)
        #stats.pop("difficulty", None) # remove difficulty from stats

        if len(stats) and stats["tp"].any():
            # here the metrics are calculated
            self.metrics.process(**stats) # class OBBMetricsCustom, 'tp', 'conf', 'pred_cls', 'target_cls', 'target_img', 'difficulty'
    
        return self.metrics.results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        #print('class DetectionValidatorCustom: print_results() called')

        # DEBUG
        #print('len metrics.keys: ', len(self.metrics.keys))
        #print('type metrics.keys: ', type(self.metrics.keys))
        #print(self.metrics.keys)

        if hasattr(self, 'target_difficulty'):
            if self.target_difficulty is None:
                LOGGER.info("\n--- Check all difficulty categories ---")
            else:
                difficulty_name = {1: "Easy", 2: "Moderate", 3: "Hard"}.get(self.target_difficulty, "Unknown")
                LOGGER.info(f"\n--- Check difficulty category: {difficulty_name} ---")
        else:
            LOGGER.info("\n--- Check all difficulty categories (standard) ---")

        pf = "%22s" + "%11i" * 2 + "%11.4g" * len(self.metrics.keys)  # metrics.keys: p, r, ap50, ap70
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))

        # DEBUG
        #print('len mean_results: ', len(self.metrics.mean_results()))
        #print('type mean_results: ', type(self.metrics.mean_results()))
        #print(self.metrics.mean_results())

        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                #  Car       3343      13783       4398      0.869      0.844      0.896      0.885
                #  Ped        817       2267       5929      0.537      0.381      0.413      0.377
                #  Cyc        602        857       3599      0.614      0.426      0.477      0.421
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )
            # DEBUG
            #print('len class_result: ', len(self.metrics.class_result(i)))
            #print('type class_result: ', type(self.metrics.class_result(i)))
            #print(self.metrics.class_result(i))

        LOGGER.info("\nDistribution of difficulty levels across all objects:")
        pf = "%22s" + "%11s" * 5  
        LOGGER.info(pf % ("Difficulty", "All", "Easy", "Moderate", "Hard", "Unknown"))
        LOGGER.info(pf % ("", self.nd_per_class.sum(), self.nd_per_class[1], self.nd_per_class[2], self.nd_per_class[3],self.nd_per_class[0]))

        LOGGER.info("\nObjects per class and difficulty level:")
        pf = "%22s" + "%11s" * 4  # Class, Easy, Moderate, Hard, Unknown
        LOGGER.info(pf % ("Class", "Easy", "Moderate", "Hard", "Unknown"))
        for class_name, difficulty_counts in self.objects_per_class_and_difficulty.items():
                LOGGER.info(
                    pf % (class_name, difficulty_counts["Easy"], difficulty_counts["Moderate"], difficulty_counts["Hard"], difficulty_counts["Unknown"])
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        #print('class DetectionValidatorCustom: build_dataset() called')
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        #print('class DetectionValidatorCustom: get_dataloader() called')
        #print('-> contains difficulty metadata')
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results
  
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
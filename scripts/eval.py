"""
Evaluation script for RT-DETR on KITTI processed data.
Computes simple mAP metrics and compares fine-tuned vs base model.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

CLASS_NAMES = {
	0: "Vehicle",
	1: "Pedestrian",
	2: "Cyclist",
}

COCO_TO_KITTI = {
	2: 0,   # car -> Vehicle
	7: 0,   # truck -> Vehicle
	5: 0,   # bus -> Vehicle
	6: 0,   # train -> Vehicle
	0: 1,   # person -> Pedestrian
	1: 2,   # bicycle -> Cyclist
	3: 2,   # motorcycle -> Cyclist
}


def _xyxy_iou(box_a, box_b):
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0.0, inter_x2 - inter_x1)
	inter_h = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area
	return inter_area / union if union > 0 else 0.0


def _compute_ap(recalls, precisions):
	if len(recalls) == 0:
		return 0.0
	mrec = np.concatenate(([0.0], recalls, [1.0]))
	mpre = np.concatenate(([0.0], precisions, [0.0]))
	for i in range(len(mpre) - 1, 0, -1):
		mpre[i - 1] = max(mpre[i - 1], mpre[i])
	idx = np.where(mrec[1:] != mrec[:-1])[0]
	ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
	return ap


def _load_checkpoint_path(checkpoint_path: str | None) -> Path:
	if checkpoint_path:
		return Path(checkpoint_path)
	checkpoints = list(RESULTS_DIR.glob("checkpoint-*"))
	if not checkpoints:
		raise FileNotFoundError(f"No checkpoints found in {RESULTS_DIR}")
	return max(checkpoints, key=lambda x: int(x.name.split("-")[1]))


def _load_model_and_processor(use_pretrained: bool, checkpoint_path: str | None):
	processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
	if use_pretrained:
		model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
		return model, processor

	ckpt = _load_checkpoint_path(checkpoint_path)
	model = RTDetrForObjectDetection.from_pretrained(ckpt)
	return model, processor


def _map_coco_to_kitti(cls_id: int) -> int | None:
	return COCO_TO_KITTI.get(cls_id)


def evaluate_dataset(
	model,
	processor,
	split_dir: Path,
	top_k_per_image: int = 100,
	iou_thresholds: list | None = None,
	max_samples: int | None = None,
	device: str | None = None,
	use_coco_mapping: bool = False
):
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	if iou_thresholds is None:
		iou_thresholds = [0.5, 0.75]

	images_dir = split_dir / "images"
	annotations_file = split_dir / "annotations.json"
	with open(annotations_file, "r") as f:
		annotations = json.load(f)

	image_ids = list(annotations.keys())
	if max_samples is not None:
		image_ids = image_ids[:max_samples]

	model = model.to(device)
	model.eval()

	preds_by_class = {c: [] for c in CLASS_NAMES.keys()}
	gts_by_image_class = {}

	for image_id in image_ids:
		image_path = images_dir / image_id
		image = Image.open(image_path).convert("RGB")

		ann = annotations[image_id]
		gts_by_image_class[image_id] = {c: [] for c in CLASS_NAMES.keys()}
		for label, box in zip(ann["labels"], ann["boxes"]):
			gts_by_image_class[image_id][label].append(box)

		inputs = processor(images=image, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			outputs = model(**inputs)

		logits = outputs.logits[0]
		scores = torch.sigmoid(logits)
		max_scores, max_classes = scores.max(dim=-1)

		top_k = min(top_k_per_image, max_scores.numel())
		topk_scores, topk_indices = max_scores.topk(top_k)
		topk_classes = max_classes[topk_indices]
		topk_boxes = outputs.pred_boxes[0][topk_indices]

		cx, cy, w, h = topk_boxes.unbind(-1)
		boxes_xyxy = torch.stack([
			(cx - 0.5 * w).cpu(),
			(cy - 0.5 * h).cpu(),
			(cx + 0.5 * w).cpu(),
			(cy + 0.5 * h).cpu(),
		], dim=-1).numpy()

		for score, cls, box in zip(topk_scores.cpu().numpy(), topk_classes.cpu().numpy(), boxes_xyxy):
			cls_id = int(cls)
			if use_coco_mapping:
				cls_id = _map_coco_to_kitti(cls_id)
				if cls_id is None:
					continue

			if cls_id in preds_by_class:
				preds_by_class[cls_id].append({
					"image_id": image_id,
					"score": float(score),
					"box": box.tolist(),
				})

	metrics = {}
	map_values = []
	for thr in iou_thresholds:
		aps = []
		for cls_id in CLASS_NAMES.keys():
			preds = sorted(preds_by_class[cls_id], key=lambda x: x["score"], reverse=True)
			npos = sum(len(gts_by_image_class[img_id][cls_id]) for img_id in gts_by_image_class)
			if npos == 0:
				continue

			tp = np.zeros(len(preds))
			fp = np.zeros(len(preds))
			used = {img_id: np.zeros(len(gts_by_image_class[img_id][cls_id])) for img_id in gts_by_image_class}

			for i, pred in enumerate(preds):
				img_id = pred["image_id"]
				gt_boxes = gts_by_image_class[img_id][cls_id]
				if len(gt_boxes) == 0:
					fp[i] = 1
					continue

				ious = [
					_xyxy_iou(pred["box"], gt_box)
					for gt_box in gt_boxes
				]
				best_iou = max(ious) if ious else 0.0
				best_idx = int(np.argmax(ious)) if ious else -1

				if best_iou >= thr and used[img_id][best_idx] == 0:
					tp[i] = 1
					used[img_id][best_idx] = 1
				else:
					fp[i] = 1

			tp_cum = np.cumsum(tp)
			fp_cum = np.cumsum(fp)
			recalls = tp_cum / max(npos, 1)
			precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
			ap = _compute_ap(recalls, precisions)
			aps.append(ap)

		map_values.append(np.mean(aps) if aps else 0.0)

	metrics["mAP@0.5"] = map_values[0] if map_values else 0.0
	if len(iou_thresholds) > 1:
		metrics["mAP@0.5:0.95"] = float(np.mean(map_values))

	return metrics


def _print_metrics(title: str, metrics: dict):
	print(f"\n{title}")
	for k, v in metrics.items():
		print(f"  {k}: {v:.4f}")


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Evaluate RT-DETR on KITTI processed data")
	parser.add_argument("--checkpoint", type=str, default=None,
						help="Path to fine-tuned checkpoint (defaults to latest)")
	parser.add_argument("--max-samples", type=int, default=None,
						help="Limit number of validation images for evaluation")
	parser.add_argument("--top-k", type=int, default=100,
						help="Top-k predictions per image for evaluation")
	parser.add_argument("--iou", type=float, nargs="*", default=[0.5, 0.75, 0.9],
						help="IoU thresholds for mAP")
	parser.add_argument("--no-base", action="store_true",
						help="Skip base model evaluation")

	args = parser.parse_args()

	split_dir = PROCESSED_DATA_DIR / "val"

	model, processor = _load_model_and_processor(
		use_pretrained=False,
		checkpoint_path=args.checkpoint
	)
	ft_metrics = evaluate_dataset(
		model,
		processor,
		split_dir=split_dir,
		top_k_per_image=args.top_k,
		iou_thresholds=args.iou,
		max_samples=args.max_samples,
		use_coco_mapping=False
	)

	_print_metrics("Fine-tuned model metrics", ft_metrics)

	if not args.no_base:
		base_model, base_processor = _load_model_and_processor(
			use_pretrained=True,
			checkpoint_path=None
		)
		base_metrics = evaluate_dataset(
			base_model,
			base_processor,
			split_dir=split_dir,
			top_k_per_image=args.top_k,
			iou_thresholds=args.iou,
			max_samples=args.max_samples,
			use_coco_mapping=True
		)
		_print_metrics("Base model (COCO) metrics", base_metrics)


if __name__ == "__main__":
	main()

"""
Run RT-DETR on a video and save predictions to reports/.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VIDEOS_DIR = PROJECT_DIR / "data" / "videos"
REPORTS_DIR = PROJECT_DIR / "reports"
RESULTS_DIR = PROJECT_DIR / "results"

CLASS_NAMES = {
	0: "Vehicle",
	1: "Pedestrian",
	2: "Cyclist",
}

CLASS_COLORS = {
	0: (0, 0, 255),
	1: (255, 0, 0),
	2: (0, 165, 255),
}


def _find_latest_checkpoint(results_dir: Path) -> Path:
	checkpoints = list(results_dir.glob("checkpoint-*"))
	if not checkpoints:
		raise FileNotFoundError(f"No checkpoints found in {results_dir}")
	return max(checkpoints, key=lambda x: int(x.name.split("-")[1]))


def _pick_video(video_path: str | None) -> Path:
	if video_path:
		return Path(video_path)

	if not VIDEOS_DIR.exists():
		raise FileNotFoundError(f"Videos folder not found: {VIDEOS_DIR}")

	candidates = []
	for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
		candidates.extend(VIDEOS_DIR.glob(ext))

	if not candidates:
		raise FileNotFoundError(f"No videos found in {VIDEOS_DIR}")

	return candidates[0]


def _score_logits(logits: torch.Tensor, use_focal_loss: bool) -> tuple[torch.Tensor, torch.Tensor]:
	if use_focal_loss:
		scores = torch.sigmoid(logits)
	else:
		scores = torch.softmax(logits, dim=-1)

	max_scores, max_classes = scores.max(dim=-1)
	return max_scores, max_classes


def _display_percent(score: float) -> float:
	# Map 0.04 -> 0.80 and 0.12 -> 1.00, cap to [0, 1]
	low_score = 0.04
	high_score = 0.12
	if score <= low_score:
		return max(0.0, (score / low_score) * 0.8)
	if score >= high_score:
		return 1.0
	return 0.8 + ((score - low_score) / (high_score - low_score)) * 0.2


def _draw_boxes(
	frame_bgr: np.ndarray,
	boxes_xyxy: np.ndarray,
	scores: np.ndarray,
	labels: np.ndarray,
	threshold: float,
	show_score: bool
):
	for box, score, label in zip(boxes_xyxy, scores, labels):
		if score < threshold:
			continue

		label_id = int(label)
		class_name = CLASS_NAMES.get(label_id, str(label_id))
		color = CLASS_COLORS.get(label_id, (0, 255, 255))

		x1, y1, x2, y2 = [int(v) for v in box]
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(frame_bgr.shape[1], x2)
		y2 = min(frame_bgr.shape[0], y2)

		cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
		if show_score:
			percent = _display_percent(float(score)) * 100.0
			label_text = f"{class_name}: {percent:.0f}%"
		else:
			label_text = class_name
		cv2.putText(frame_bgr, label_text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
	parser = argparse.ArgumentParser(description="Run RT-DETR on a video")
	parser.add_argument("--video", type=str, default=None, help="Path to input video (default: first in data/videos)")
	parser.add_argument("--threshold", type=float, default=0.01, help="Score threshold for predictions")
	parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: latest in results)")
	parser.add_argument("--output", type=str, default=None, help="Output video path (default: reports/<name>_pred.mp4)")
	parser.add_argument("--show-score", action="store_true", help="Overlay confidence score on boxes")
	args = parser.parse_args()

	video_path = _pick_video(args.video)
	checkpoint_path = Path(args.checkpoint) if args.checkpoint else _find_latest_checkpoint(RESULTS_DIR)

	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	output_path = Path(args.output) if args.output else REPORTS_DIR / f"{video_path.stem}_pred.mp4"

	processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
	model = RTDetrForObjectDetection.from_pretrained(checkpoint_path)
	model.eval()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = model.to(device)
	use_focal_loss = bool(getattr(model.config, "use_focal_loss", True))

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 30
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

	frame_count = 0
	while True:
		ret, frame_bgr = cap.read()
		if not ret:
			break

		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(frame_rgb)
		inputs = processor(images=pil_image, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = model(**inputs)

		logits = outputs.logits[0]
		scores, labels = _score_logits(logits, use_focal_loss)
		boxes = outputs.pred_boxes[0]

		cx, cy, w, h = boxes.unbind(-1)
		boxes_xyxy = torch.stack([
			(cx - 0.5 * w) * width,
			(cy - 0.5 * h) * height,
			(cx + 0.5 * w) * width,
			(cy + 0.5 * h) * height,
		], dim=-1).cpu().numpy()

		_draw_boxes(
			frame_bgr,
			boxes_xyxy,
			scores.cpu().numpy(),
			labels.cpu().numpy(),
			args.threshold,
			args.show_score
		)
		writer.write(frame_bgr)

		frame_count += 1
		if frame_count % 50 == 0:
			print(f"Processed {frame_count} frames...")

	cap.release()
	writer.release()

	print(f"Saved output video to: {output_path}")
	print(f"Use focal loss: {use_focal_loss}, threshold: {args.threshold}")


if __name__ == "__main__":
	main()

"""
Error Analysis Script for RT-DETR Object Detection
Visualizes model predictions vs ground truth on validation samples
"""
import random
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

# Class names (3 classes: Vehicle, Pedestrian, Cyclist)
CLASS_NAMES = {
    0: "Vehicle",
    1: "Pedestrian", 
    2: "Cyclist"
}

# Colors for visualization
GROUND_TRUTH_COLOR = "green"
PREDICTION_COLORS = {
    0: "red",       # Vehicle
    1: "blue",      # Pedestrian
    2: "orange"     # Cyclist
}


def load_model_and_processor(checkpoint_path: Path = None, use_pretrained: bool = False):
    """Load the trained RT-DETR model and image processor.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint
        use_pretrained: If True, use original COCO-pretrained model with class mapping
    """
    # Load processor
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    
    if use_pretrained:
        print("Using original COCO-pretrained model with class mapping")
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        model.eval()
        return model, processor, True  # Return flag indicating pretrained mode
    
    if checkpoint_path is None:
        # Find the best checkpoint
        checkpoints = list(RESULTS_DIR.glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {RESULTS_DIR}")
        # Get latest checkpoint (highest number)
        checkpoint_path = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    
    print(f"Loading model weights from checkpoint: {checkpoint_path.name}")
    print(f"  Full path: {checkpoint_path}")
    
    # Load model
    model = RTDetrForObjectDetection.from_pretrained(checkpoint_path)
    model.eval()
    
    return model, processor, False  # Return flag indicating fine-tuned mode


# COCO class IDs to our class mapping
# Vehicle (0): car, truck, bus, train
# Pedestrian (1): person
# Cyclist (2): bicycle, motorcycle
COCO_TO_KITTI = {
    2: 0,   # car -> Vehicle
    7: 0,   # truck -> Vehicle
    5: 0,   # bus -> Vehicle
    6: 0,   # train -> Vehicle
    0: 1,   # person -> Pedestrian
    1: 2,   # bicycle -> Cyclist
    3: 2,   # motorcycle -> Cyclist
}


def load_validation_sample(idx: int = None):
    """
    Load a sample from the validation dataset.
    
    Args:
        idx: Optional index. If None, selects a random sample.
        
    Returns:
        image: PIL Image
        annotations: Dict with 'labels' and 'boxes'
        image_path: Path to the image
    """
    val_dir = PROCESSED_DATA_DIR / "val"
    images_dir = val_dir / "images"
    annotations_file = val_dir / "annotations.json"
    
    # Load annotations
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    
    # Get image list
    image_ids = list(annotations.keys())
    
    # Select sample
    if idx is None:
        idx = random.randint(0, len(image_ids) - 1)
    
    image_id = image_ids[idx]
    image_path = images_dir / image_id
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    print(f"Loaded sample {idx}: {image_id}")
    print(f"  Ground truth objects: {len(annotations[image_id]['labels'])}")
    
    return image, annotations[image_id], image_path


def run_inference(model, processor, image: Image.Image, device: str = None):
    """
    Run model inference on an image.
    
    Args:
        model: RT-DETR model
        processor: RT-DETR image processor
        image: PIL Image
        device: Device to run on (auto-detected if None)
        
    Returns:
        outputs: Model outputs with logits and pred_boxes
        original_size: Tuple of (height, width) of original image
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs, (image.height, image.width)


def decode_predictions(outputs, processor, original_size, threshold: float = 0.5, use_coco_mapping: bool = False, top_k: int = None):
    """
    Decode model outputs to predictions using the official post-processing.
    
    Args:
        outputs: Model outputs from RT-DETR
        processor: RT-DETR image processor for post-processing
        original_size: Tuple of (height, width) of original image
        threshold: Minimum probability threshold (after sigmoid for focal loss)
        use_coco_mapping: If True, map COCO classes to KITTI classes
        
    Returns:
        predictions: List of dicts with 'box', 'label', 'confidence'
    """
    # Debug: Print raw output shapes and values
    print(f"  Debug - Logits shape: {outputs.logits.shape}")
    print(f"  Debug - Pred boxes shape: {outputs.pred_boxes.shape}")
    
    # Check raw logit values
    logits = outputs.logits[0]  # [num_queries, num_classes]
    print(f"  Debug - Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
    
    # Apply SIGMOID for focal loss (RT-DETR default with use_focal_loss=True)
    import torch.nn.functional as F
    scores = torch.sigmoid(logits)
    print(f"  Debug - Scores after sigmoid min: {scores.min().item():.4f}, max: {scores.max().item():.4f}")
    
    # Get top scores per query (max probability for each query)
    max_scores, max_classes = scores.max(dim=-1)
    top_k_scores, top_k_indices = max_scores.topk(min(10, len(max_scores)))
    print(f"  Debug - Top 10 max scores: {[f'{s:.4f}' for s in top_k_scores.tolist()]}")
    print(f"  Debug - Top 10 predicted classes: {[max_classes[i].item() for i in top_k_indices]}")
    
    # Manual post-processing using sigmoid scores
    # Filter predictions by threshold (use 0.005-0.01 for focal loss sigmoid)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, max_scores.numel())
        topk_scores, topk_indices = max_scores.topk(top_k)
        filtered_boxes = outputs.pred_boxes[0][topk_indices]
        filtered_scores = topk_scores
        filtered_classes = max_classes[topk_indices]
    else:
        mask = max_scores >= threshold
        filtered_boxes = outputs.pred_boxes[0][mask]
        filtered_scores = max_scores[mask]
        filtered_classes = max_classes[mask]
    
    # Convert boxes from cxcywh (normalized) to xyxy (pixel coordinates)
    img_h, img_w = original_size
    boxes_cxcywh = filtered_boxes.cpu()
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    boxes_xyxy = torch.stack([
        (cx - 0.5 * w) * img_w,  # x_min
        (cy - 0.5 * h) * img_h,  # y_min
        (cx + 0.5 * w) * img_w,  # x_max
        (cy + 0.5 * h) * img_h   # y_max
    ], dim=-1)
    
    # Create results dict
    results = {
        'boxes': boxes_xyxy,
        'scores': filtered_scores.cpu(),
        'labels': filtered_classes.cpu()
    }
    
    print(f"  Debug - Manual post-process returned {len(results['scores'])} detections")
    if len(results['scores']) > 0:
        print(f"  Debug - Result scores: {[f'{s.item():.4f}' for s in results['scores'][:10]]}")
    
    predictions = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_id = label.item()
        
        # Map COCO class to KITTI class if using pretrained model
        if use_coco_mapping:
            if label_id not in COCO_TO_KITTI:
                continue  # Skip classes we don't care about
            label_id = COCO_TO_KITTI[label_id]
        
        # Box is in absolute coordinates [x_min, y_min, x_max, y_max]
        # Convert to normalized coordinates for consistency with ground truth
        h, w = original_size
        normalized_box = [
            box[0].item() / w,  # x_min
            box[1].item() / h,  # y_min
            box[2].item() / w,  # x_max
            box[3].item() / h   # y_max
        ]
        
        predictions.append({
            "box": normalized_box,
            "label": label_id,
            "confidence": score.item()
        })
    
    if top_k is not None and top_k > 0:
        print(f"  Predictions (top_k={top_k}): {len(predictions)} objects")
    else:
        print(f"  Predictions (threshold={threshold}): {len(predictions)} objects")
    
    return predictions




def visualize_comparison(
    image: Image.Image,
    ground_truth: dict,
    predictions: list,
    threshold: float = 0.5,
    save_path: Path = None,
    show: bool = True
):
    """
    Visualize ground truth boxes and predicted boxes on the image.
    
    Args:
        image: PIL Image
        ground_truth: Dict with 'labels' and 'boxes' (normalized xyxy)
        predictions: List of prediction dicts with 'box', 'label', 'confidence'
        threshold: Confidence threshold used for filtering predictions
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    img_width, img_height = image.size
    img_array = np.array(image)
    
    # ===== Left plot: Ground Truth =====
    ax1 = axes[0]
    ax1.imshow(img_array)
    ax1.set_title("Ground Truth", fontsize=16, fontweight="bold", color="green")
    
    gt_boxes = ground_truth["boxes"]
    gt_labels = ground_truth["labels"]
    
    for box, label in zip(gt_boxes, gt_labels):
        # Box is normalized [x_min, y_min, x_max, y_max]
        x_min = box[0] * img_width
        y_min = box[1] * img_height
        x_max = box[2] * img_width
        y_max = box[3] * img_height
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=3, edgecolor=GROUND_TRUTH_COLOR, facecolor="none"
        )
        ax1.add_patch(rect)
        
        # Add label
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        ax1.text(
            x_min, y_min - 5,
            class_name,
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=GROUND_TRUTH_COLOR, alpha=0.8)
        )
    
    ax1.axis("off")
    
    # ===== Right plot: Predictions =====
    ax2 = axes[1]
    ax2.imshow(img_array)
    ax2.set_title(f"Model Predictions (conf > {threshold})", fontsize=16, fontweight="bold", color="red")
    
    for pred in predictions:
        box = pred["box"]
        label = pred["label"]
        confidence = pred["confidence"]
        
        # Box is normalized [x_min, y_min, x_max, y_max]
        x_min = box[0] * img_width
        y_min = box[1] * img_height
        x_max = box[2] * img_width
        y_max = box[3] * img_height
        
        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Get color for this class
        color = PREDICTION_COLORS.get(label, "yellow")
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=3, edgecolor=color, facecolor="none"
        )
        ax2.add_patch(rect)
        
        # Add label with confidence
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        label_text = f"{class_name}: {confidence:.2f}"
        ax2.text(
            x_min, y_min - 5,
            label_text,
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
        )
    
    ax2.axis("off")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def analyze_sample(
    idx: int = None,
    threshold: float = 0.5,
    save_path: Path = None,
    show: bool = True,
    checkpoint_path: Path = None,
    use_pretrained: bool = False,
    top_k: int = None
):
    """
    Main function to analyze a single validation sample.
    
    Args:
        idx: Sample index (random if None)
        threshold: Confidence threshold for predictions
        save_path: Optional path to save visualization
        show: Whether to display the plot
        checkpoint_path: Optional path to specific checkpoint
        use_pretrained: If True, use COCO-pretrained model with class mapping
    """
    print("=" * 60)
    print("RT-DETR Error Analysis")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, processor, is_pretrained = load_model_and_processor(checkpoint_path, use_pretrained)
    
    # Load sample
    print("\n[2/4] Loading validation sample...")
    image, annotations, image_path = load_validation_sample(idx)
    
    # Run inference
    print("\n[3/4] Running inference...")
    outputs, original_size = run_inference(model, processor, image)
    
    # Decode predictions using official post-processing
    predictions = decode_predictions(
        outputs,
        processor,
        original_size,
        threshold=threshold,
        use_coco_mapping=is_pretrained,
        top_k=top_k
    )
    
    # Print comparison summary
    print("\n" + "-" * 40)
    print("Summary:")
    print(f"  Ground truth objects: {len(annotations['labels'])}")
    print(f"  Predicted objects: {len(predictions)}")
    
    # Print ground truth details
    print("\nGround Truth:")
    for i, (label, box) in enumerate(zip(annotations["labels"], annotations["boxes"])):
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        print(f"  {i+1}. {class_name}: box={[f'{b:.3f}' for b in box]}")
    
    # Print prediction details
    print("\nPredictions:")
    if predictions:
        for i, pred in enumerate(predictions):
            class_name = CLASS_NAMES.get(pred["label"], f"Class {pred['label']}")
            box_str = [f"{b:.3f}" for b in pred["box"]]
            print(f"  {i+1}. {class_name}: conf={pred['confidence']:.3f}, box={box_str}")
    else:
        print("  No predictions above threshold")
    
    print("-" * 40)
    
    # Visualize
    print("\n[4/4] Generating visualization...")
    visualize_comparison(image, annotations, predictions, threshold=threshold, save_path=save_path, show=show)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return {
        "image_path": str(image_path),
        "ground_truth": annotations,
        "predictions": predictions
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Error analysis for RT-DETR object detection")
    parser.add_argument("--idx", type=int, default=None,
                        help="Sample index (random if not specified)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--top-k", type=int, default=None,
                        help="If set, return top-k predictions per image (overrides threshold)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save visualization")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display the plot")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use COCO-pretrained model with class mapping instead of fine-tuned")
    
    args = parser.parse_args()
    
    # Run analysis
    result = analyze_sample(
        idx=args.idx,
        threshold=args.threshold,
        save_path=Path(args.save) if args.save else None,
        show=not args.no_show,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        use_pretrained=args.pretrained,
        top_k=args.top_k
    )

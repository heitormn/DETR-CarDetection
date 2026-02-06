"""
Utility functions for KITTI dataset visualization and processing
"""
import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import torch 
from torch.utils.data import Dataset
import albumentations as A


class KITTIObject:
    """KITTI object annotation"""
    def __init__(self, label_line):
        data = label_line.strip().split(' ')
        self.type = data[0]  # Object class
        self.truncated = float(data[1])
        self.occluded = int(data[2])
        self.alpha = float(data[3])
        # 2D bounding box
        self.bbox = [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
        # 3D dimensions
        self.dimensions = [float(data[8]), float(data[9]), float(data[10])]
        # 3D location
        self.location = [float(data[11]), float(data[12]), float(data[13])]
        self.rotation_y = float(data[14])


def load_kitti_sample(image_path, label_path):
    """
    Load a KITTI sample (image + annotations)
    
    Args:
        image_path: Path to the image file
        label_path: Path to the label file
        
    Returns:
        image: PIL Image
        objects: List of KITTIObject
    """
    # Load image
    image = Image.open(image_path)
    
    # Load annotations
    objects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    objects.append(KITTIObject(line))
    
    return image, objects


def plot_kitti_samples(image_paths, label_paths, save_dir=None):
    """
    Visualize KITTI samples with bounding boxes
    
    Args:
        image_paths: List of paths to image files
        label_paths: List of paths to label files (must match image_paths order)
        save_dir: Optional directory to save visualizations. If None, displays interactively.
    
    Example:
        >>> image_paths = ['../data/raw/training/image_2/000000.png', '../data/raw/training/image_2/000001.png']
        >>> label_paths = ['../data/raw/training/label_2/000000.txt', '../data/raw/training/label_2/000001.txt']
        >>> plot_kitti_samples(image_paths, label_paths)
    """
    if len(image_paths) != len(label_paths):
        raise ValueError(f"Number of images ({len(image_paths)}) must match number of labels ({len(label_paths)})")
    
    if len(image_paths) == 0:
        print("No images provided.")
        return
    
    print(f"Visualizing {len(image_paths)} samples from KITTI dataset...")
    
    # Color map for different classes
    color_map = {
        'Car': 'red',
        'Van': 'orange',
        'Truck': 'brown',
        'Pedestrian': 'green',
        'Person_sitting': 'lime',
        'Cyclist': 'blue',
        'Tram': 'purple',
        'Misc': 'gray',
        'DontCare': 'lightgray'
    }
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize each sample
    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        image_path = Path(image_path)
        label_path = Path(label_path)
        
        # Load sample
        image, objects = load_kitti_sample(image_path, label_path)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        for obj in objects:
            if obj.type == 'DontCare':
                continue
                
            bbox = obj.bbox  # [left, top, right, bottom]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Get color for this class
            color = color_map.get(obj.type, 'yellow')
            
            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{obj.type}"
            ax.text(
                bbox[0], bbox[1] - 5,
                label,
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
            )
        
        # Set title and remove axes
        ax.set_title(f"Sample {idx + 1}: {image_path.name} - {len(objects)} objects", 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save or show
        if save_dir:
            output_file = save_path / f"sample_{idx + 1}_{image_path.stem}.png"
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            print(f"Saved: {output_file}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    print(f"\nVisualization complete! Processed {len(image_paths)} samples.")
    
    # Print class statistics
    print("\n" + "="*60)
    print("Class statistics:")
    class_counts = {}
    for image_path, label_path in zip(image_paths, label_paths):
        _, objects = load_kitti_sample(image_path, label_path)
        for obj in objects:
            if obj.type != 'DontCare':
                class_counts[obj.type] = class_counts.get(obj.type, 0) + 1
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    print("="*60)

def get_kitti_samples(data_dir='../data/raw', k=5, split='training'):
    """
    Get paths to the first k samples from KITTI dataset
    
    Args:
        data_dir: Path to the KITTI dataset directory
        k: Number of samples to get
        split: 'training' or 'testing'
        
    Returns:
        image_paths: List of image file paths
        label_paths: List of label file paths
    """
    data_path = Path(data_dir)
    image_dir = data_path / split / 'image_2'
    label_dir = data_path / split / 'label_2'
    
    # Get image files
    image_files = sorted(list(image_dir.glob('*.png')))[:k]
    
    # Get corresponding label files
    image_paths = [str(f) for f in image_files]
    label_paths = [str(label_dir / f"{f.stem}.txt") for f in image_files]
    
    return image_paths, label_paths


# KITTI class mapping for RT-DETR
# Simplified to 3 classes: Vehicle (0), Pedestrian (1), Cyclist (2)
KITTI_CLASSES = {
    'Car': 0,
    'Van': 0,
    'Truck': 0,
    'Tram': 0,
    'Pedestrian': 1,
    'Person_sitting': 1,
    'Cyclist': 2
    # 'Misc' and 'DontCare' are excluded
}

# Preprocessing parameters for RT-DETR
PREPROCESS_CONFIG = {
    'target_size': (640, 640),
    'image_mean': [0.485, 0.456, 0.406],
    'image_std': [0.229, 0.224, 0.225]
}



class KITTIDetectionDataset(Dataset):
    """KITTI dataset for RT-DETR object detection with optional augmentation."""
    
    def __init__(self, split_dir: Path, image_processor, augment=True):
        """
        Args:
            split_dir: Path to split directory (e.g., data/processed/train)
            image_processor: RTDetrImageProcessor instance
            augment: Whether to apply data augmentation
        """
        
        self.split_dir = split_dir
        self.images_dir = split_dir / 'images'
        self.image_processor = image_processor
        self.augment = augment
        
        # Load annotations
        annotations_file = split_dir / 'annotations.json'
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image paths
        self.image_ids = list(self.annotations.keys())
        self.image_paths = [str(self.images_dir / img_id) for img_id in self.image_ids]
        
        # Define augmentations (only for training)
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussNoise(var_range=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.2),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]
        boxes = np.array(ann['boxes'], dtype=np.float32)  # Normalized [x_min, y_min, x_max, y_max]
        labels = np.array(ann['labels'], dtype=np.int64)
        
        # Apply augmentations
        if self.augment and self.transform is not None and len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels.tolist()
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['class_labels'], dtype=np.int64)
        
        # Convert back to PIL for processor
        image_pil = Image.fromarray(image)
        img_width, img_height = image_pil.size
        
        # Prepare annotations in COCO format for RT-DETR processor
        # Processor expects pixel coordinates in [x, y, width, height] format
        # Our boxes are currently [x_min, y_min, x_max, y_max] normalized (0-1)
        target = {
            'image_id': idx,
            'annotations': [
                {
                    # Convert from normalized [x_min, y_min, x_max, y_max] to pixel [x, y, width, height]
                    'bbox': [
                        box[0] * img_width,                              # x (top-left x in pixels)
                        box[1] * img_height,                             # y (top-left y in pixels)
                        (box[2] - box[0]) * img_width,                   # width in pixels
                        (box[3] - box[1]) * img_height                   # height in pixels
                    ],
                    'category_id': int(label),
                    'area': float((box[2] - box[0]) * (box[3] - box[1]) * img_width * img_height),  # Area in pixels
                    'iscrowd': 0
                }
                for box, label in zip(boxes, labels)
            ]
        }
        
        # Process with RT-DETR image processor
        encoding = self.image_processor(
            images=image_pil,
            annotations=target,
            return_tensors='pt'
        )
        
        # Remove batch dimension from tensors only (labels might be lists)
        encoding = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}
        
        return encoding


def process_kitti_for_rtdetr(
    raw_dir: str = '../data/raw',
    processed_dir: str = '../data/processed',
    train_val_split: float = 0.2,
    seed: int = 42
):
    """
    Process KITTI dataset for RT-DETR training.
    
    Structure:
    - raw/training/ -> split into processed/train/ and processed/val/
    - raw/testing/ -> becomes processed/test/
    
    Each processed split contains:
    - images/: image files
    - annotations.json: COCO-style annotations with labels and boxes
    
    Args:
        raw_dir: Path to raw KITTI data
        processed_dir: Path to save processed data
        train_val_split: Fraction of training data to use for validation (0-1)
        seed: Random seed for reproducible splits
        
    Returns:
        Dict with statistics about the processed dataset
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Set random seed
    np.random.seed(seed)
    
    stats = {
        'train': {'images': 0, 'objects': 0, 'class_counts': {}},
        'val': {'images': 0, 'objects': 0, 'class_counts': {}},
        'test': {'images': 0, 'objects': 0, 'class_counts': {}}
    }
    
    print("="*70)
    print("Processing KITTI dataset for RT-DETR")
    print("="*70)
    
    # Process training split (split into train and val)
    train_image_dir = raw_path / 'training' / 'image_2'
    train_label_dir = raw_path / 'training' / 'label_2'
    
    if train_image_dir.exists():
        print(f"\n[1/2] Processing training data from {train_image_dir}...")
        
        # Get all image files
        image_files = sorted(list(train_image_dir.glob('*.png')))
        n_images = len(image_files)
        
        if n_images == 0:
            print("  ⚠️  No images found in training directory!")
        else:
            # Shuffle and split
            indices = np.arange(n_images)
            np.random.shuffle(indices)
            
            n_val = int(n_images * train_val_split)
            val_indices = set(indices[:n_val])
            train_indices = set(indices[n_val:])
            
            print(f"  Found {n_images} images")
            print(f"  Split: {len(train_indices)} train, {len(val_indices)} val")
            
            # Process train split
            _process_split(
                image_files, train_label_dir, train_indices,
                processed_path / 'train', stats['train']
            )
            
            # Process val split
            _process_split(
                image_files, train_label_dir, val_indices,
                processed_path / 'val', stats['val']
            )
    else:
        print(f"\n⚠️  Training directory not found: {train_image_dir}")
    
    # Process testing split
    test_image_dir = raw_path / 'testing' / 'image_2'
    test_label_dir = raw_path / 'testing' / 'label_2'
    
    if test_image_dir.exists():
        print(f"\n[2/2] Processing testing data from {test_image_dir}...")
        
        test_image_files = sorted(list(test_image_dir.glob('*.png')))
        n_test = len(test_image_files)
        
        if n_test == 0:
            print("  ⚠️  No images found in testing directory!")
        else:
            print(f"  Found {n_test} images")
            test_indices = set(range(n_test))
            
            _process_split(
                test_image_files, test_label_dir, test_indices,
                processed_path / 'test', stats['test']
            )
    else:
        print(f"\n⚠️  Testing directory not found: {test_image_dir}")
    
    # Save dataset statistics
    stats_file = processed_path / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
    _print_stats(stats)
    print(f"\nStatistics saved to: {stats_file}")
    
    return stats


def _process_split(
    image_files: List[Path],
    label_dir: Path,
    indices: set,
    output_dir: Path,
    stats: Dict
):
    """
    Process a single split (train/val/test) and save in RT-DETR format.
    
    Creates:
    - output_dir/images/: copied images
    - output_dir/annotations.json: dict mapping image_id -> {'labels': [...], 'boxes': [...]}
    """
    # Create output directories
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = {}
    target_width, target_height = PREPROCESS_CONFIG['target_size']
    
    for idx in indices:
        image_file = image_files[idx]
        label_file = label_dir / f"{image_file.stem}.txt"
        
        # Load image
        image = Image.open(image_file)
        orig_width, orig_height = image.size
        
        # Resize to target size
        image_resized = image.resize((target_width, target_height), Image.LANCZOS)
        
        # Save resized image
        dest_image = images_dir / image_file.name
        image_resized.save(dest_image)
        
        # Parse annotations
        labels = []
        boxes = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    obj = KITTIObject(line)
                    
                    # Skip DontCare regions
                    if obj.type == 'DontCare':
                        continue
                    
                    # Skip unknown classes
                    if obj.type not in KITTI_CLASSES:
                        continue
                    
                    # Get class label
                    class_id = KITTI_CLASSES[obj.type]
                    labels.append(class_id)
                    
                    # Normalize bounding box coordinates (x_min, y_min, x_max, y_max)
                    # RT-DETR expects normalized coordinates [0, 1]
                    bbox = obj.bbox  # [left, top, right, bottom]
                    normalized_box = [
                        bbox[0] / orig_width,   # x_min
                        bbox[1] / orig_height,  # y_min
                        bbox[2] / orig_width,   # x_max
                        bbox[3] / orig_height   # y_max
                    ]
                    boxes.append(normalized_box)
                    
                    # Update stats
                    stats['class_counts'][obj.type] = stats['class_counts'].get(obj.type, 0) + 1
        
        # Store annotation
        annotations[image_file.name] = {
            'labels': labels,
            'boxes': boxes,
            'image_width': target_width,
            'image_height': target_height
        }
        
        # Update stats
        stats['images'] += 1
        stats['objects'] += len(labels)
    
    # Save annotations
    ann_file = output_dir / 'annotations.json'
    with open(ann_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Save preprocessing config
    config_file = output_dir / 'preprocess_config.json'
    with open(config_file, 'w') as f:
        json.dump(PREPROCESS_CONFIG, f, indent=2)
    
    print(f"  ✓ Processed {stats['images']} images -> {output_dir}")
    print(f"    Resized to {PREPROCESS_CONFIG['target_size']}")
    print(f"    Objects: {stats['objects']}, Annotations saved to {ann_file.name}")


def _print_stats(stats: Dict):
    """Print dataset statistics."""
    for split_name, split_stats in stats.items():
        if split_stats['images'] == 0:
            continue
            
        print(f"\n{split_name.upper()} split:")
        print(f"  Images: {split_stats['images']}")
        print(f"  Objects: {split_stats['objects']}")
        
        if split_stats['class_counts']:
            print(f"  Class distribution:")
            for cls, count in sorted(split_stats['class_counts'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"    {cls}: {count}")


def load_processed_split(processed_dir: str, split: str = 'train') -> Tuple[List[str], Dict]:
    """
    Load a processed split for training/evaluation.
    
    Args:
        processed_dir: Path to processed data directory
        split: 'train', 'val', or 'test'
        
    Returns:
        image_paths: List of image file paths
        annotations: Dict mapping image_id -> {'labels': [...], 'boxes': [...]}
    """
    split_dir = Path(processed_dir) / split
    images_dir = split_dir / 'images'
    ann_file = split_dir / 'annotations.json'
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Get image paths
    image_paths = [str(images_dir / img_id) for img_id in annotations.keys()]
    
    return image_paths, annotations


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize KITTI dataset samples')
    parser.add_argument('--data_dir', type=str, default='../data/raw',
                        help='Path to KITTI dataset directory')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations (if not provided, will display)')
    parser.add_argument('--process', action='store_true',
                        help='Process KITTI data for RT-DETR training')
    parser.add_argument('--processed_dir', type=str, default='../data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--train_val_split', type=float, default=0.2,
                        help='Fraction of training data for validation')
    
    args = parser.parse_args()
    
    if args.process:
        # Process data
        process_kitti_for_rtdetr(
            raw_dir=args.data_dir,
            processed_dir=args.processed_dir,
            train_val_split=args.train_val_split
        )
    else:
        # Get sample paths and visualize
        image_paths, label_paths = get_kitti_samples(args.data_dir, args.k)
        plot_kitti_samples(image_paths, label_paths)
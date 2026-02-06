# Data Directory

This directory contains the datasets used for training and evaluation.

## Directory Structure

```
data/
├── raw/          # Raw KITTI dataset (ignored by git)
├── processed/    # Preprocessed data for training (ignored by git)
└── README.md     # This file
```

## Downloading the KITTI Object Detection Dataset

### Using the Download Script

Run the provided download script from the scripts directory:

```bash
cd scripts
python download_kitti.py
```

This will automatically download and extract:
- **Training images** (7,481 images)
- **Training labels** (7,481 annotation files)
- **Testing images** (7,518 images)

The data will be organized as:
```
data/raw/
├── training/
│   ├── image_2/  # Training images
│   └── label_2/  # Training annotations
└── testing/
    └── image_2/  # Testing images (no labels available)
```

### Manual Download

Alternatively, you can manually download from the KITTI website:

1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
2. Download:
   - Left color images of object data set (12 GB)
   - Training labels of object data set (5 MB)
3. Extract both zip files into `data/raw/`

### Dataset Information

- **Classes**: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
- **Image size**: Variable (typically around 1242×375 pixels)
- **Format**: 
  - Images: PNG
  - Labels: TXT (one file per image with space-separated values)

### Label Format

Each line in a label file represents one object with the following fields:
```
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom dimensions_height dimensions_width dimensions_length location_x location_y location_z rotation_y
```

For 2D object detection, the main fields of interest are:
- `type`: Object class (Car, Pedestrian, etc.)
- `bbox_left, bbox_top, bbox_right, bbox_bottom`: 2D bounding box coordinates

## Notes

- The `raw/` and `processed/` directories are ignored by git due to their large size
- Ensure you have at least 15 GB of free disk space before downloading
- Download may take some time depending on your internet connection

## Data Preprocessing

After downloading the raw KITTI dataset, you need to preprocess it for RT-DETR training.

### Class Mapping

We simplify the 8 KITTI classes to **3 classes**:

| KITTI Original | Our Class | Class ID |
|---------------|-----------|----------|
| Car, Van, Truck, Tram | **Vehicle** | 0 |
| Pedestrian, Person_sitting | **Pedestrian** | 1 |
| Cyclist | **Cyclist** | 2 |

*Note: `Misc` and `DontCare` annotations are excluded from training.*

### Preprocessing Steps

The preprocessing pipeline performs:

1. **Image Resizing**: All images resized to **640×640 pixels** using LANCZOS resampling
2. **Normalization Parameters**: 
   - Mean: `[0.485, 0.456, 0.406]` (ImageNet statistics)
   - Std: `[0.229, 0.224, 0.225]`
3. **Bounding Box Normalization**: Boxes converted to normalized coordinates `[x_min, y_min, x_max, y_max]` in range `[0, 1]`
4. **Data Splitting**:
   - `raw/training/` → `processed/train/` (80%) + `processed/val/` (20%)
   - `raw/testing/` → `processed/test/` (100%)

### Running Preprocessing

```bash
cd scripts
python utils.py --process
```

**Options:**
```bash
# Custom validation split (default: 0.2 = 20%)
python utils.py --process --train_val_split 0.15

# Custom paths
python utils.py --process --data_dir ../data/raw --processed_dir ../data/processed
```

### Processed Data Structure

After preprocessing, the `processed/` directory will contain:

```
processed/
├── train/
│   ├── images/                    # Resized 640×640 images
│   ├── annotations.json           # Image annotations in RT-DETR format
│   └── preprocess_config.json     # Normalization parameters
├── val/
│   ├── images/
│   ├── annotations.json
│   └── preprocess_config.json
├── test/
│   ├── images/
│   ├── annotations.json
│   └── preprocess_config.json
└── dataset_stats.json             # Overall dataset statistics
```

### Annotations Format

The `annotations.json` file contains RT-DETR compatible annotations:

```json
{
  "000001.png": {
    "labels": [0, 0, 1],           // Class IDs
    "boxes": [                      // Normalized [x_min, y_min, x_max, y_max]
      [0.123, 0.456, 0.789, 0.890],
      [0.234, 0.345, 0.567, 0.678],
      [0.890, 0.123, 0.987, 0.456]
    ],
    "image_width": 640,
    "image_height": 640
  }
}
```

### Dataset Statistics

Check `processed/dataset_stats.json` for:
- Number of images per split (train/val/test)
- Total object counts per split
- Class distribution across splits

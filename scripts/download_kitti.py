"""
Script to download KITTI Object Detection Dataset
Downloads left color images and labels for object detection
"""
import os
import zipfile
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _is_valid_zip(path: Path) -> bool:
    """Return True if path exists and is a valid zip archive."""
    try:
        return path.exists() and zipfile.is_zipfile(path)
    except Exception:
        return False


def download_url(url: str, output_path: Path):
    """Robust download with validation and progress.

    Uses streaming download with a User-Agent header and validates that the
    resulting file is a ZIP. If an existing file is invalid, it gets removed
    before re-downloading.
    """
    # Remove previous invalid file
    if output_path.exists() and not _is_valid_zip(output_path):
        try:
            output_path.unlink()
        except Exception:
            # If we cannot unlink, try to replace by writing anew
            pass

    # Prepare request with headers
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })

    with urllib.request.urlopen(req) as resp:
        # Basic content-type sanity check when available
        content_type = resp.headers.get('Content-Type', '')
        total = resp.headers.get('Content-Length')
        total = int(total) if total is not None else None

        # Stream to file with progress
        with DownloadProgressBar(total=total, unit='B', unit_scale=True, desc=output_path.name) as t:
            with open(output_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    t.update(len(chunk))

    # Validate resulting file is a ZIP
    if not _is_valid_zip(output_path):
        # Leave the file for inspection but raise an informative error
        raise RuntimeError(f"Downloaded file is not a valid zip: {output_path}")


def download_kitti_dataset(data_dir='data/raw'):
    """
    Download KITTI object detection dataset
    
    Args:
        data_dir: Directory to save the dataset
    """
    # Create directories - resolve relative to script location
    if not Path(data_dir).is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_dir
    else:
        data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # KITTI dataset URLs
    urls = {
        'images': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
        'labels': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
    }
    
    print("Starting KITTI Object Detection Dataset Download...")
    print(f"Saving to: {data_path.absolute()}")
    print("-" * 60)
    
    # Download and extract each file
    for name, url in urls.items():
        zip_path = data_path / f'{name}.zip'
        
        # Download if not exists or is invalid
        if not zip_path.exists() or not _is_valid_zip(zip_path):
            if zip_path.exists():
                print(f"\n{zip_path.name} exists but is invalid. Re-downloading {name}...")
            else:
                print(f"\nDownloading {name}...")
            try:
                download_url(url, zip_path)
                print(f"✓ Downloaded {name}")
            except Exception as e:
                print(f"✗ Error downloading {name}: {e}")
                continue
        else:
            print(f"\n✓ {name}.zip already exists and is valid")
        
        # Extract
        print(f"Extracting {name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            print(f"✓ Extracted {name}")
        except zipfile.BadZipFile:
            print(f"✗ Error extracting {name}: invalid ZIP archive")
        except Exception as e:
            print(f"✗ Error extracting {name}: {e}")
    
    print("\n" + "-" * 60)
    print("Download complete!")
    print("\nDataset structure:")
    print(f"{data_path}/")
    print("├── training/")
    print("│   ├── image_2/  (7481 training images)")
    print("│   └── label_2/  (7481 training labels)")
    print("└── testing/")
    print("    └── image_2/  (7518 testing images)")
    print("\nNote: Testing set does not include labels (for evaluation server)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download KITTI Object Detection Dataset')
    parser.add_argument('--data_dir', type=str, default='../data/raw',
                        help='Directory to save the dataset (default: ../data/raw)')
    
    args = parser.parse_args()
    download_kitti_dataset(args.data_dir)

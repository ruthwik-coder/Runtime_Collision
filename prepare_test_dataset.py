"""
prepare_test_dataset.py
-------------------------
Lightweight Dataset Preparation & Ingestion Script (Capped at 1-2 GB Total).

Directory Structure:
Test_Dataset/
├── 1_Clear_Accidents/        (Ground Truth: Accident=True, 100-300MB)
├── 2_Blurry_CCTV/             (Ground Truth: Accident=True, 100-300MB)
├── 3_Hard_Negatives/          (Ground Truth: Accident=False, 60MB Highway Traffic)
└── 4_Severe_MultiVehicle/     (Ground Truth: Accident=True, 100-300MB)
"""

import os
import sys
import json
import glob
import shutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_ROOT = "Test_Dataset"
MAX_FILES_PER_CATEGORY = 10  # Capped to keep dataset lightweight (~1-2 GB max)
MAX_TOTAL_SIZE_MB = 1500      # 1.5 GB total limit

CATEGORIES = {
    "1_Clear_Accidents": {
        "ground_truth_accident": True,
        "description": "High visibility accident CCTV/Dashcam clips",
        "source_url": "https://www.kaggle.com/datasets/picekl/accident"
    },
    "2_Blurry_CCTV": {
        "ground_truth_accident": True,
        "description": "Low resolution, blurry, night, or rainy weather accident footage",
        "source_url": "https://www.kaggle.com/datasets/picekl/accident"
    },
    "3_Hard_Negatives": {
        "ground_truth_accident": False,
        "description": "Normal highway traffic, heavy congestion, lane merges (NO ACCIDENTS)",
        "source_url": "https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset/data"
    },
    "4_Severe_MultiVehicle": {
        "ground_truth_accident": True,
        "description": "Multi-car pileups, truck/bus involvement, severe highway collisions",
        "source_url": "https://www.kaggle.com/datasets/picekl/accident"
    }
}

def create_folders():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    for cat in CATEGORIES.keys():
        path = os.path.join(DATASET_ROOT, cat)
        os.makedirs(path, exist_ok=True)
        print(f"[+] Directory ready: {path}")

def get_folder_size_mb(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def download_lightweight_datasets():
    """Downloads public Kaggle datasets under size cap."""
    try:
        import kagglehub
        print("\n[*] Ingesting Kaggle Datasets (Capped at 1-2 GB)...")
        
        # 1. Hard Negatives (60.5 MB)
        path_hn = os.path.join(DATASET_ROOT, "3_Hard_Negatives")
        if len(os.listdir(path_hn)) == 0:
            print("  -> Downloading aryashah2k/highway-traffic-videos-dataset for 3_Hard_Negatives (60MB)...")
            path_download = kagglehub.dataset_download("aryashah2k/highway-traffic-videos-dataset")
            copy_videos(path_download, path_hn, max_files=MAX_FILES_PER_CATEGORY)
        else:
            print(f"  [✓] 3_Hard_Negatives already populated ({len(os.listdir(path_hn))} files).")
            
    except Exception as e:
        print(f"[!] Notice during Kaggle download: {e}")

def copy_videos(src_dir, target_dir, max_files=10):
    """Utility to copy video files into target folder up to max_files limit."""
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    found_files = []
    for ext in video_extensions:
        found_files.extend(glob.glob(os.path.join(src_dir, "**", ext), recursive=True))
    
    copied_count = 0
    for file_path in found_files[:max_files]:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(target_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy2(file_path, dest_path)
            copied_count += 1
    print(f"     Copied {copied_count} videos to {target_dir}")

def generate_manifest():
    """Scans Test_Dataset directory and creates dataset_manifest.json."""
    manifest = {
        "dataset_name": "Highway Accident Detection Test Benchmark (Lightweight 1-2 GB Edition)",
        "categories": {}
    }
    
    total_videos = 0
    total_mb = get_folder_size_mb(DATASET_ROOT)
    
    for cat_name, meta in CATEGORIES.items():
        cat_path = os.path.join(DATASET_ROOT, cat_name)
        video_files = []
        if os.path.exists(cat_path):
            for file in os.listdir(cat_path):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    video_files.append(file)
        
        manifest["categories"][cat_name] = {
            "description": meta["description"],
            "ground_truth_accident": meta["ground_truth_accident"],
            "source_url": meta["source_url"],
            "video_count": len(video_files),
            "videos": video_files
        }
        total_videos += len(video_files)
    
    manifest["total_test_videos"] = total_videos
    manifest["total_dataset_size_mb"] = round(total_mb, 2)
    
    manifest_path = os.path.join(DATASET_ROOT, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[+] Manifest updated: {manifest_path}")
    print(f"    Total Videos: {total_videos} | Total Size: {total_mb:.2f} MB (~{total_mb/1024:.2f} GB)")

if __name__ == "__main__":
    print("=" * 60)
    print("LIGHTWEIGHT TEST DATASET BUILDER (1-2 GB LIMIT)")
    print("=" * 60)
    create_folders()
    download_lightweight_datasets()
    generate_manifest()

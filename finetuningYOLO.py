import os
import yaml
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Path to the folder containing 'train', 'valid', 'test' folders
DATASET_ROOT = "./datasets/face_mask"
YAML_PATH = "face_mask_data.yaml"
MODEL_WEIGHTS = "yolov8n.pt"


# =================================================

def organize_dataset():
    """
    YOLOv8 strict requirement:
    Images must be in /images/train
    Labels must be in /labels/train

    If your dataset has mixed images and txt files in the same folder,
    this function fixes it.
    """
    print("--- Checking Dataset Structure ---")

    # We expect DATASET_ROOT to contain 'train', 'valid' (or 'val'), 'test'
    # Check what subfolders exist
    subsets = ['train', 'valid', 'test', 'val']

    found_mixed = False

    for subset in subsets:
        # Check standard mixed path: dataset_root/train
        mixed_path = os.path.join(DATASET_ROOT, subset)

        # Check if it might be inside an 'images' folder already but mixed: dataset_root/images/train
        mixed_images_path = os.path.join(DATASET_ROOT, 'images', subset)

        target_dir = mixed_path if os.path.exists(mixed_path) else mixed_images_path

        if os.path.exists(target_dir):
            # Look for mixed content
            images = glob.glob(os.path.join(target_dir, "*.jpg")) + glob.glob(os.path.join(target_dir, "*.png"))
            labels = glob.glob(os.path.join(target_dir, "*.txt"))

            if len(images) > 0 and len(labels) > 0:
                found_mixed = True
                print(f"⚠️ Found mixed images and labels in {subset}. separating...")

                # Define new paths
                new_image_dir = os.path.join(DATASET_ROOT, "images", subset)
                new_label_dir = os.path.join(DATASET_ROOT, "labels", subset)

                os.makedirs(new_image_dir, exist_ok=True)
                os.makedirs(new_label_dir, exist_ok=True)

                # Move files
                for img in images:
                    shutil.move(img, os.path.join(new_image_dir, os.path.basename(img)))
                for lbl in labels:
                    # Don't move classes.txt if it exists, just copy it or ignore
                    if "classes.txt" in lbl:
                        continue
                    shutil.move(lbl, os.path.join(new_label_dir, os.path.basename(lbl)))

                # Remove empty old dir if it's different from new dirs
                if target_dir != new_image_dir and target_dir != new_label_dir:
                    try:
                        os.rmdir(target_dir)
                    except:
                        pass

    if found_mixed:
        print("✅ Dataset organized into /images and /labels structure.")
    else:
        print("ℹ️ Dataset structure looks correct (or empty).")


def get_dataset_info():
    """
    Scans the label files to determine number of classes.
    """
    # Look for label files in the standard location
    label_dir = os.path.join(DATASET_ROOT, "labels", "train")

    # If not found there, maybe they are in 'valid' or just 'labels'
    if not os.path.exists(label_dir):
        label_dir = os.path.join(DATASET_ROOT, "labels")

    txt_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)

    if not txt_files:
        print(f"ERROR: No .txt label files found in {label_dir}")
        return None, None

    print(f"Scanning {len(txt_files)} label files for classes...")

    max_id = 0
    class_ids = set()

    for txt_file in txt_files[:500]:  # Check up to 500 files
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        class_ids.add(cid)
                        if cid > max_id:
                            max_id = cid
                    except ValueError:
                        continue

    num_classes = max_id + 1
    # Try to provide helpful names based on common mask datasets
    # Usually: 0=Mask, 1=No Mask, 2=Incorrect (if 3 classes)
    if num_classes == 2:
        names = ['mask', 'no_mask']
    elif num_classes == 3:
        names = ['mask', 'no_mask', 'incorrect_mask']
    else:
        names = [f"class_{i}" for i in range(num_classes)]

    print(f"ℹ️ Auto-detected {num_classes} classes: {names}")
    return num_classes, names


def create_yolo_yaml():
    organize_dataset()  # Fix structure first

    nc, names = get_dataset_info()
    if nc is None:
        return False

    # Check for train images location
    train_path = os.path.join(DATASET_ROOT, "images", "train")
    if not os.path.exists(train_path):
        # Fallback check
        train_path = os.path.join(DATASET_ROOT, "images")

    data_yaml = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'images/train',
        'val': 'images/valid',  # Ensure your folder is named 'valid' or 'val'
        'test': 'images/test',
        'nc': nc,
        'names': names
    }

    with open(YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ Generated YOLO config at: {YAML_PATH}")
    return True


def plot_metrics(save_dir):
    csv_path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box')
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box')
    plt.title('Localization Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class')
    plt.title('Classification Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
    plt.close()


def train_yolo():
    if not create_yolo_yaml():
        return

    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=YAML_PATH,
        epochs=10,
        imgsz=640,
        batch=16,
        patience=5,
        name="yolo_face_mask_finetune",
        device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )

    print(f"Best model: {results.save_dir}/weights/best.pt")
    plot_metrics(results.save_dir)


if __name__ == "__main__":
    train_yolo()
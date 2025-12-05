import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt  # <--- NEW IMPORT
from tqdm import tqdm
from Models import MobileFaceNet, Arcface

# ==========================================
# PART 1: CONFIGURATION
# ==========================================
CONFIG = {
    "DATA_PATH": "./AFDB_face_dataset/AFDB_face_dataset",
    "WEIGHTS_PATH": "model_mobilefacenet.pth",
    "SAVE_PATH": "afdb_finetuned.pth",
    "PLOT_PATH": "training_curves.png",  # <--- NEW CONFIG
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "LR_BACKBONE": 1e-4,
    "LR_HEAD": 1e-2,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


# ==========================================
# PART 2: DATASET (Unchanged)
# ==========================================
class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Valid extensions
        exts = ('.jpg', '.jpeg', '.png', '.bmp')

        if not os.path.exists(root_dir):
            raise RuntimeError(f"Dataset path not found: {root_dir}")

        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        print(f"Dataset initialized. Found {len(self.class_names)} identities.")

        for cls_name in self.class_names:
            class_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(exts):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))


class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# ==========================================
# PART 3: UTILS & PLOTTING
# ==========================================
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001, path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, backbone, head):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, backbone, head)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, backbone, head)
            self.counter = 0

    def save_checkpoint(self, val_loss, backbone, head):
        torch.save({'backbone': backbone.state_dict(), 'head': head.state_dict()}, self.path)
        self.val_loss_min = val_loss


# <--- NEW FUNCTION: Plots history
def plot_training_results(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")
    # plt.show() # Uncomment if running in Jupyter/Colab


# ==========================================
# PART 4: TRAINING LOOP
# ==========================================
def train():
    print(f"Running on Device: {CONFIG['DEVICE']}")

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load Data
    full_dataset = CustomFaceDataset(CONFIG['DATA_PATH'], transform=None)
    num_classes = len(full_dataset.class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(TransformedSubset(train_subset, train_transforms),
                              batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    val_loader = DataLoader(TransformedSubset(val_subset, val_transforms),
                            batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

    # Initialize Model
    backbone = MobileFaceNet(embedding_size=512).to(CONFIG['DEVICE'])
    head = Arcface(embedding_size=512, classnum=num_classes, s=30.0, m=0.35).to(CONFIG['DEVICE'])

    # Load Weights
    if os.path.exists(CONFIG['WEIGHTS_PATH']):
        state_dict = torch.load(CONFIG['WEIGHTS_PATH'], map_location=CONFIG['DEVICE'])
        if 'backbone' in state_dict:
            backbone.load_state_dict(state_dict['backbone'], strict=False)
        else:
            backbone.load_state_dict(state_dict, strict=False)
        print("Weights loaded.")
    else:
        print("Pretrained weights not found, training from scratch.")

    # Freeze Layers
    frozen_layers = ['conv1', 'conv2_dw', 'conv_23', 'conv_3']
    for name, child in backbone.named_children():
        if name in frozen_layers:
            for param in child.parameters(): param.requires_grad = False

    optimizer = optim.SGD([
        {'params': filter(lambda p: p.requires_grad, backbone.parameters()), 'lr': CONFIG['LR_BACKBONE']},
        {'params': head.parameters(), 'lr': CONFIG['LR_HEAD']}
    ], momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=6, path=CONFIG['SAVE_PATH'])

    # <--- HISTORY TRACKING
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStarting training...")
    for epoch in range(CONFIG['EPOCHS']):
        backbone.train()
        head.train()

        run_loss, correct, total = 0.0, 0, 0

        for i, (img, label) in enumerate(tqdm(train_loader, total=len(train_loader), leave=True)):
            img, label = img.to(CONFIG['DEVICE']), label.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            emb = backbone(img)
            output = head(emb, label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_acc = 100 * correct / total
        avg_train_loss = run_loss / len(train_loader)

        # Validation
        backbone.eval()
        head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(CONFIG['DEVICE']), label.to(CONFIG['DEVICE'])
                emb = backbone(img)
                output = head(emb, label)  # Arcface head used for val loss
                loss = criterion(output, label)
                val_loss += loss.item()
                _, pred = torch.max(output, 1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Ep {epoch + 1}: T-Loss {avg_train_loss:.4f} Acc {train_acc:.1f}% | V-Loss {avg_val_loss:.4f} Acc {val_acc:.1f}%")

        scheduler.step(avg_val_loss)
        early_stop(avg_val_loss, backbone, head)

        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    # <--- PLOT RESULTS AT END
    plot_training_results(history, CONFIG['PLOT_PATH'])


if __name__ == "__main__":
    train()
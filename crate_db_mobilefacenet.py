import torch
import numpy as np
import os
import pickle
import cv2
import sys
from ultralytics import YOLO

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


from model import MobileFaceNet
from Yolo import run_face_inference_train


DB_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\db'
SAVE_FILE = 'face_db.pkl'
WEIGHTS_PATH = 'model_mobilefacenet.pth'
EMBEDDING_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Rulez pe: {device}")


class RecursiveImageDataset(Dataset):
    def __init__(self, directory):
        self.directories_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.directories_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.directories_list)

    def __getitem__(self, idx):
        path = self.directories_list[idx]
        # Citire BGR cu OpenCV
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return np.zeros((100, 100, 3), dtype=np.uint8), path
        return img_bgr, path


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(full_image_bgr, coords, model):

    if coords is None:
        return None
    try:
        x1, y1, x2, y2 = coords
    except ValueError:
        return None

    h, w = full_image_bgr.shape[:2]


    pad_w = int((x2 - x1) * 0.10)
    pad_h = int((y2 - y1) * 0.10)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)


    if x2 - x1 < 10 or y2 - y1 < 10: return None

    face_crop = full_image_bgr[y1:y2, x1:x2]

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)


    input_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()[0]

    return embedding


print("Încarc modelele")
try:
    det_model = YOLO('yolov12m-face.pt')
except:
    print("❌ Nu găsesc yolov12m-face.pt!")
    sys.exit()



embed_model = MobileFaceNet(EMBEDDING_SIZE).to(device)
try:
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        if 'state_dict' in checkpoint:
            embed_model.load_state_dict(checkpoint['state_dict'])
        else:
            embed_model.load_state_dict(checkpoint)
        embed_model.eval()
        print("MobileFaceNet Loaded!")
    else:
        print(f"Nu gasesc {WEIGHTS_PATH}")
        sys.exit()
except Exception as e:
    print(f"Eroare MobileFaceNet: {e}")
    sys.exit()

# Start Procesare
dataset = RecursiveImageDataset(DB_PATH)
temp_db = {}

print(f"\n Încep procesarea a {len(dataset)} imagini ")
print("-" * 60)

for i in range(len(dataset)):
    img_bgr, full_path = dataset[i]


    if np.sum(img_bgr) == 0: continue


    path_parts = full_path.split(os.sep)
    person_folder = None
    for part in path_parts:
        if " - " in part:
            person_folder = part
            break

    if person_folder is None:
        continue


    clean_name = person_folder.split('-')[0].strip()


    coords = run_face_inference_train(det_model, img_bgr)


    if coords is None:
        continue

    vector = get_embedding(img_bgr, coords, embed_model)

    if vector is not None:
        if clean_name not in temp_db:
            temp_db[clean_name] = []
        temp_db[clean_name].append(vector)
        print(f"\r✅ Procesat: {clean_name:<15} | Folder: {person_folder}", end="")

print(f"\n\n{'-' * 60}")
print("⚗️  Unific vectorii si Normalizez...")

final_db = {}
for name, vector_list in temp_db.items():
    if len(vector_list) > 0:
        # 1. Facem media tuturor ipostazelor persoanei
        avg_vector = np.mean(vector_list, axis=0)

        # 2. NORMALIZARE CRITICA (Vectorul final trebuie sa aiba lungimea 1)
        avg_vector = avg_vector / np.linalg.norm(avg_vector)

        final_db[name] = avg_vector
        print(f"  👤 {name}: DB salvat ({len(vector_list)} imagini sursa).")

# Salvare
with open(SAVE_FILE, 'wb') as f:
    pickle.dump(final_db, f)

print(f"\n✅ Succes! Baza de date '{SAVE_FILE}' este gata de testare.")
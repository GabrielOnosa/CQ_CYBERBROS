import torch
import numpy as np
import os
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from ultralytics import YOLO
import sys

# --- IMPORTURILE TALE ---
from model import MobileFaceNet
from Yolo import run_face_inference_train

# ==========================================
# 0. CONFIGURĂRI
# ==========================================
DB_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\db'
SAVE_FILE = 'face_db.pkl'
EMBEDDING_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Rulez pe: {device}")


# ==========================================
# 1. CLASE ȘI FUNCȚII
# ==========================================
class RecursiveImageDataset(Dataset):
    def __init__(self, directory):
        self.directories_list = []
        # Parcurgem recursiv toate folderele
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Salvăm calea completă
                    self.directories_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.directories_list)

    def __getitem__(self, idx):
        path = self.directories_list[idx]
        image_array = Image.open(path).convert('RGB')
        return np.array(image_array), path


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(full_image_arr, coords, model):
    if coords is None or coords[0] is None:
        return None
    x1, y1, x2, y2 = coords
    h, w = full_image_arr.shape[:2]

    # Safe Crop
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 10 or y2 - y1 < 10: return None

    face_crop = full_image_arr[y1:y2, x1:x2]
    input_tensor = preprocess(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_tensor)
        embedding = embedding.cpu().numpy()[0]
    return embedding


# ==========================================
# 2. PROCESARE
# ==========================================
# Incarcare Modele
print("⏳ Încarc modelele...")
try:
    det_model = YOLO('yolov12m-face.pt')
except:
    print("❌ Nu găsesc yolov12m-face.pt!")
    sys.exit()

embed_model = MobileFaceNet(EMBEDDING_SIZE).to(device)
try:
    checkpoint = torch.load('model_mobilefacenet.pth', map_location=device)
    if 'state_dict' in checkpoint:
        embed_model.load_state_dict(checkpoint['state_dict'])
    else:
        embed_model.load_state_dict(checkpoint)
    embed_model.eval()
except Exception as e:
    print(f"❌ Eroare MobileFaceNet: {e}")
    sys.exit()

# Start Procesare
dataset = RecursiveImageDataset(DB_PATH)
temp_db = {}

print(f"\n🚀 Încep procesarea a {len(dataset)} imagini...")
print("-" * 50)

for i in range(len(dataset)):
    img_array, full_path = dataset[i]

    # --- LOGICA NOUĂ DE EXTRAGERE NUME ---
    # Spargem calea în bucăți: ['db', 'indoor_persons', 'Diego - Indoor', 'img.jpg']
    path_parts = full_path.split(os.sep)

    person_folder = None
    # Căutăm exact folderul care conține " - " (cratimă)
    for part in path_parts:
        if " - " in part:
            person_folder = part
            break

    if person_folder is None:
        # Dacă nu găsim un folder cu cratimă, sărim peste (e probabil fișier rătăcit)
        # print(f"⚠️ Ignor fișierul (nu respectă formatul Nume - Locație): {full_path}")
        continue

    # Curățăm numele: "Diego - Indoor" -> "Diego"
    clean_name = person_folder.split('-')[0].strip()

    # --- DETECTIE & EMBEDDING ---
    coords = run_face_inference_train(det_model, img_array)
    vector = get_embedding(img_array, coords, embed_model)

    if vector is not None:
        if clean_name not in temp_db:
            temp_db[clean_name] = []
        temp_db[clean_name].append(vector)
        # Afișăm un mic progres (suprascrie linia curentă ca să nu umple consola)
        print(f"\r✅ Procesat: {clean_name:<15} | Sursa: {person_folder}", end="")

print(f"\n\n{'-' * 50}")
print("⚗️  Calculez media vectorilor (Unificare Indoor + Outdoor)...")

final_db = {}
for name, vector_list in temp_db.items():
    if len(vector_list) > 0:
        avg_vector = np.mean(vector_list, axis=0)
        # Normalizare esențială
        avg_vector = avg_vector / np.linalg.norm(avg_vector)
        final_db[name] = avg_vector
        print(f"  👤 {name}: Bază de date creată din {len(vector_list)} imagini.")

# Salvare
with open(SAVE_FILE, 'wb') as f:
    pickle.dump(final_db, f)

print(f"\n✅ Succes! Baza de date '{SAVE_FILE}' conține {len(final_db)} persoane unice.")
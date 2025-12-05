import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pickle
import os
import sys
from torchvision import transforms
from PIL import Image

# --- IMPORTURI CUSTOM ---
from model import MobileFaceNet
from Yolo import run_face_inference_test  # Varianta complexă pentru test

# ==========================================
# 0. CONFIGURĂRI
# ==========================================
# Calea specifică către imaginea ta
TEST_IMAGE_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\test\Outdoor\Masked\Pablo - Outdoor - M12C.png'
#TEST_IMAGE_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\test\Outdoor\Non-masked\Pablo - Outdoor - 6C.png'
# Fișierele necesare
DB_FILE = 'face_db.pkl'
WEIGHTS_PATH = 'model_mobilefacenet.pth'

# Configurare detecție
THRESHOLD = 0.13 # Pragul de siguranță (40%)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️  System ready using: {device}")

# ==========================================
# 1. PREGĂTIRE (HELPER FUNCTIONS)
# ==========================================

# Preprocesare identică cu cea de la antrenare
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(full_image_bgr, coords, model):
    """ Extrage vectorul dintr-o față decupată """
    x1, y1, x2, y2 = coords
    h, w = full_image_bgr.shape[:2]

    # Safe Crop (să nu ieșim din poză)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Dacă fața e prea mică, o ignorăm
    if x2 - x1 < 10 or y2 - y1 < 10: return None

    # Decupare din imaginea mare
    face_crop = full_image_bgr[y1:y2, x1:x2]

    # !!! CRITIC: Conversie BGR -> RGB !!!
    # OpenCV citește BGR, MobileFaceNet vrea RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # Transformare în Tensor și mutare pe GPU
    input_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_tensor)
        embedding = embedding.cpu().numpy()[0]

    return embedding


# ==========================================
# 2. ÎNCĂRCARE RESURSE
# ==========================================

# A. Baza de date
if not os.path.exists(DB_FILE):
    print(f"❌ EROARE: Nu găsesc '{DB_FILE}'. Rulează create_db.py!")
    sys.exit()

print("📂 Încarc baza de date...")
with open(DB_FILE, 'rb') as f:
    face_db = pickle.load(f)
print(f"✅ Bază de date încărcată. Conține {len(face_db)} persoane.")

# B. Modele YOLO
print("⏳ Încarc YOLO...")
try:
    person_model = YOLO('yolo11n.pt')  # Detectează oameni
    face_model = YOLO('yolov12m-face.pt')  # Detectează fețe
except Exception as e:
    print(f"❌ Eroare YOLO: {e}")
    sys.exit()

# C. MobileFaceNet
print("⏳ Încarc MobileFaceNet...")
embed_model = MobileFaceNet(512).to(device)
try:
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if 'state_dict' in checkpoint:
        embed_model.load_state_dict(checkpoint['state_dict'])
    else:
        embed_model.load_state_dict(checkpoint)
    embed_model.eval()
except Exception as e:
    print(f"❌ Eroare MobileFaceNet: {e}")
    sys.exit()

# ==========================================
# 3. EXECUȚIE PE IMAGINEA TA
# ==========================================

print(f"\n🖼️  Procesez imaginea: {os.path.basename(TEST_IMAGE_PATH)}")

# 1. Citire Imagine
frame = cv2.imread(TEST_IMAGE_PATH)
if frame is None:
    print("❌ Nu am putut citi imaginea! Verifică calea.")
    sys.exit()

# 2. Detecție Complexă (Person -> Upscale -> Face)
detections = run_face_inference_test(person_model, face_model, frame)

if len(detections) == 0:
    print("⚠️  Nu am detectat nicio față în imagine.")
    # Afișăm imaginea originală oricum
    cv2.imshow("Rezultat", frame)
    cv2.waitKey(0)
    sys.exit()

print(f"✅ Am detectat {len(detections)} față/fețe. Încep recunoașterea...")

# 3. Recunoaștere pentru fiecare față
for i, det in enumerate(detections):
    box = det['face_box']  # [x1, y1, x2, y2]

    # Generăm amprenta feței curente (Embedding)
    current_vector = get_embedding(frame, box, embed_model)

    if current_vector is None: continue

    # Normalizăm vectorul curent (Esențial pentru acuratețe)
    current_vector = current_vector / np.linalg.norm(current_vector)

    # --- COMPARARE CU CEI 10 OAMENI DIN BAZĂ ---
    best_name = "Unknown"
    max_similarity = -1.0

    print(f"\n--- Analiză Fața #{i + 1} ---")

    for db_name, db_vector in face_db.items():
        # Calculăm similaritatea (Produs Scalar)
        score = np.dot(current_vector, db_vector)

        # Afișăm scorul pentru fiecare persoană (pentru debug)
        print(f"   vs {db_name:<10}: {score:.4f}")

        if score > max_similarity:
            max_similarity = score
            best_name = db_name

    # 4. Decizie Finală
    if max_similarity >= THRESHOLD:
        print(f"🏆 REZULTAT: Este {best_name} (Siguranță: {max_similarity:.2f})")
        color = (0, 255, 0)  # Verde
        text = f"{best_name} {max_similarity:.2f}"
    else:
        print(f"⚠️ REZULTAT: Persoană Necunoscută (Cel mai apropiat: {best_name} cu {max_similarity:.2f})")
        color = (0, 0, 255)  # Roșu
        text = f"Unknown ({best_name}?)"

    # 5. Desenare pe imagine
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cvzone.cornerRect(frame, [x1, y1, w, h], l=15, rt=1, colorR=color)
    cvzone.putTextRect(frame, text, (max(0, x1), max(20, y1 - 10)), scale=1, thickness=1, colorR=color)

# ==========================================
# 4. AFIȘARE VIZUALĂ
# ==========================================
h, w = frame.shape[:2]
# Redimensionăm doar pentru afișare dacă e prea mare
if h > 1000:
    scale = 1000 / h
    dim = (int(w * scale), 1000)
    frame = cv2.resize(frame, dim)

cv2.imshow("TEST PABLO", frame)
print("\nApasa orice tasta pe fereastra imaginii pentru a inchide...")
cv2.waitKey(0)
cv2.destroyAllWindows()
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pickle
import os
import sys
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- IMPORTURI CUSTOM ---
from model import Backbone
from Yolo import run_face_inference_test

# --- IMPORT SUPER RESOLUTION (OPTIONAL) ---
try:
    from cv2 import dnn_superres

    HAS_CV2_CONTRIB = True
except ImportError:
    HAS_CV2_CONTRIB = False

# ==========================================
# 0. CONFIGURĂRI UNICE
# ==========================================
# Calea către imaginea ta specifică
TEST_IMAGE_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\test\Outdoor\Masked\Marcos - Outdoor - M3C.png'

DB_FILE = 'face_db.pkl'
WEIGHTS_PATH = 'model_ir_se50.pth'

# Configurare Super Resolution
SR_MODEL_PATH = 'FSRCNN_x3.pb'
SR_SCALE = 3
UPSCALING_THRESHOLD = 60  # Daca fata e mai mica de atat pixeli latime, facem upscale

# Pragul de recunoastere
THRESHOLD = 0.13  #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  System ready using: {device}")

# ==========================================
# 1. SETUP MODEL SI PROCESARE
# ==========================================

# A. Super Resolution
sr = None
use_sr = False
if HAS_CV2_CONTRIB and os.path.exists(SR_MODEL_PATH):
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(SR_MODEL_PATH)
        sr.setModel("fsrcnn", SR_SCALE)
        use_sr = True
        print(f"✅ FSRCNN Activat.")
    except:
        print("⚠️ FSRCNN a dat eroare la incarcare, continui fara.")

# B. Transformari (Normalizare critica!)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # <--- ESENTIAL
])


# C. Functie Embedding
def get_embedding(full_image_bgr, coords, model):
    x1, y1, x2, y2 = coords
    h, w = full_image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, False

    face_crop_bgr = full_image_bgr[y1:y2, x1:x2]
    h_crop, w_crop = face_crop_bgr.shape[:2]

    # --- FSRCNN UPSCALE ---
    did_upscale = False
    if use_sr and w_crop < UPSCALING_THRESHOLD:
        try:
            face_crop_bgr = sr.upsample(face_crop_bgr)
            did_upscale = True
        except:
            pass

    # CONVERSIE BGR -> RGB PENTRU MODEL (OpenCV citeste BGR, PyTorch vrea RGB)
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(input_tensor)
        embedding = embedding.cpu().numpy()[0]

    return embedding, did_upscale


# ==========================================
# 2. INCARCARE RESURSE
# ==========================================
# 1. Baza de date
if not os.path.exists(DB_FILE):
    print(f"❌ EROARE: Nu gasesc {DB_FILE}")
    sys.exit()

with open(DB_FILE, 'rb') as f:
    face_db = pickle.load(f)
print(f"📚 DB Incarcat: {len(face_db)} persoane.")

# 2. Modele YOLO
print("⏳ Loading YOLO...")
person_model = YOLO('yolo12s.pt')
face_model = YOLO('yolov12m-face.pt')

# 3. Model IR-SE50
print(f"⏳ Loading IR-SE50...")
embed_model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)

if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if 'state_dict' in checkpoint:
        embed_model.load_state_dict(checkpoint['state_dict'])
    else:
        embed_model.load_state_dict(checkpoint)
    embed_model.eval()
    print("✅ IR-SE50 Ready!")
else:
    print(f"❌ Nu gasesc {WEIGHTS_PATH}")
    sys.exit()

# ==========================================
# 3. SINGLE INFERENCE
# ==========================================
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Imaginea nu exista: {TEST_IMAGE_PATH}")
    sys.exit()

# Citire Imagine (OpenCV citeste BGR)
frame = cv2.imread(TEST_IMAGE_PATH)
if frame is None:
    print("❌ Nu am putut citi imaginea (format invalid?).")
    sys.exit()

print(f"\n🖼️ Procesez imaginea: {os.path.basename(TEST_IMAGE_PATH)}")

# 1. Detectie
detections = run_face_inference_test(person_model, face_model, frame)

if len(detections) == 0:
    print("⚠️ Nicio față detectată.")
else:
    print(f"👀 Am gasit {len(detections)} fete.")

    for i, det in enumerate(detections):
        box = det['face_box']

        # Extragere Embedding
        res_emb, upscaled = get_embedding(frame, box, embed_model)

        if res_emb is None:
            continue

        # Normalizare vector (Cosine Sim cere vectori unitari)
        vector = res_emb / np.linalg.norm(res_emb)

        # Comparare
        best_name = "Unknown"
        best_score = 0.0

        for db_name, db_vector in face_db.items():
            sim = np.dot(vector, db_vector)
            if sim > best_score:
                best_score = sim
                best_name = db_name

        # Validare prag
        final_name = best_name if best_score > THRESHOLD else "Unknown"

        # Afisare consola
        tag_sr = "[Upscaled]" if upscaled else ""
        print(f"   ➤ Fata {i + 1}: {final_name} (Score: {best_score:.4f}) {tag_sr}")

        # Desenare pe imagine
        x1, y1, x2, y2 = box
        w_box, h_box = x2 - x1, y2 - y1

        color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)  # Verde vs Rosu

        cvzone.cornerRect(frame, [x1, y1, w_box, h_box], l=15, rt=1, colorR=color)

        text_disp = f"{final_name} {best_score:.2f}"
        cv2.putText(frame, text_disp, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

# Afisare Finala
h, w = frame.shape[:2]
# Resize daca e prea mare pentru ecran
if h > 800:
    frame = cv2.resize(frame, (int(w * 800 / h), 800))

cv2.imshow("Single Inference Result", frame)
print("\nApasa orice tasta pe fereastra imaginii pentru a inchide...")
cv2.waitKey(0)
cv2.destroyAllWindows()
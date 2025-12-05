import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pickle
import os
import sys
import time
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


from model import MobileFaceNet
from Yolo import run_face_inference_test


try:
    from cv2 import dnn_superres

    HAS_CV2_CONTRIB = True
except ImportError:
    HAS_CV2_CONTRIB = False
    print("\n⚠️ ATENTIE: Modulul 'dnn_superres' lipseste.")
    print("   Super Resolution va fi dezactivat automat.\n")


TEST_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\test'
DB_FILE = 'face_db.pkl'
WEIGHTS_PATH = 'model_mobilefacenet.pth'


SR_MODEL_PATH = 'FSRCNN_x3.pb'
SR_SCALE = 3
UPSCALING_THRESHOLD = 60

THRESHOLD = 0.13
VISUALIZE = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" System ready using: {device}")


print(" Initializare Super Resolution (FSRCNN)...")
use_sr = False
if HAS_CV2_CONTRIB:
    try:
        if os.path.exists(SR_MODEL_PATH):
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(SR_MODEL_PATH)
            sr.setModel("fsrcnn", SR_SCALE)
            use_sr = True
            print(f" FSRCNN Activat (Scale x{SR_SCALE})!")
        else:
            print(f"Fisierul '{SR_MODEL_PATH}' nu exista. SR dezactivat.")
    except Exception as e:
        print(f"Eroare SR: {e}")
        use_sr = False
else:
    print(" FSRCNN dezactivat.")



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
        img_bgr = cv2.imread(path)
        return img_bgr, path


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



def get_embedding(full_image_bgr, coords, model):
    # Initializare contoare
    t_preprocess = 0
    t_upscale = 0
    t_inference = 0


    if device.type == 'cuda': torch.cuda.synchronize()
    start_total = time.perf_counter()

    t0 = time.perf_counter()

    x1, y1, x2, y2 = coords
    h, w = full_image_bgr.shape[:2]

    # Padding
    pad_w = int((x2 - x1) * 0.1)
    pad_h = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, False, {}

    face_crop = full_image_bgr[y1:y2, x1:x2]
    h_crop, w_crop = face_crop.shape[:2]

    t_preprocess += (time.perf_counter() - t0)


    did_upscale = False
    if use_sr and w_crop < UPSCALING_THRESHOLD:
        t_sr_start = time.perf_counter()
        try:
            face_crop = sr.upsample(face_crop)
            did_upscale = True
        except:
            pass
        if device.type == 'cuda': torch.cuda.synchronize()
        t_upscale = time.perf_counter() - t_sr_start


    t0 = time.perf_counter()

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

    if device.type == 'cuda': torch.cuda.synchronize()
    t_preprocess += (time.perf_counter() - t0)

    t_inf_start = time.perf_counter()

    with torch.no_grad():

        embedding = model(input_tensor).cpu().numpy()[0]

    if device.type == 'cuda': torch.cuda.synchronize()
    t_inference = time.perf_counter() - t_inf_start


    total_time = time.perf_counter() - start_total

    timings = {
        "prep_ms": t_preprocess * 1000,
        "sr_ms": t_upscale * 1000,
        "infer_ms": t_inference * 1000,
        "total_ms": total_time * 1000
    }

    return embedding, did_upscale, timings


def extract_ground_truth_name(filename):
    try:
        parts = filename.split(' - ')
        return parts[0].strip()
    except:
        return "Unknown"



if not os.path.exists(DB_FILE):
    print(f"❌ EROARE: Baza de date '{DB_FILE}' nu exista!")
    sys.exit()

with open(DB_FILE, 'rb') as f:
    face_db = pickle.load(f)

print("⏳ Loading YOLO & Rec Model...")
person_model = YOLO('yolo12s.pt')
face_model = YOLO('yolov12m-face.pt')

print(f"⏳ Incarc MobileFaceNet din {WEIGHTS_PATH}...")
embed_model = MobileFaceNet(512).to(device)

try:
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if 'state_dict' in checkpoint:
        embed_model.load_state_dict(checkpoint['state_dict'])
    else:
        embed_model.load_state_dict(checkpoint)
    embed_model.eval()
    print("✅ Model MobileFaceNet Incarcat!")
except Exception as e:
    print(f"❌ Eroare la incarcare model .pth: {e}")
    sys.exit()


dataset = RecursiveImageDataset(TEST_PATH)

stats = {
    "total": 0, "correct": 0, "wrong": 0, "unknown": 0, "no_face_detected": 0,
    "indoor_nomask_correct": 0, "indoor_nomask_total": 0,
    "indoor_masked_correct": 0, "indoor_masked_total": 0,
    "outdoor_nomask_correct": 0, "outdoor_nomask_total": 0,
    "outdoor_masked_correct": 0, "outdoor_masked_total": 0
}

print(f"\n🚀 START EVALUARE PE {len(dataset)} IMAGINI (FARA TTA/FLIP)...")
print("-" * 80)
print(f"{'FILENAME':<35} | {'REAL':<10} | {'PREDICTED':<15} | {'SCORE'} | {'RES'} | {'SR'}")
print("-" * 80)

for i in range(len(dataset)):
    frame, path = dataset[i]
    if frame is None: continue

    filename = os.path.basename(path)
    real_name = extract_ground_truth_name(filename)


    is_masked = "Masked" in path and "Non-masked" not in path
    is_indoor = "Indoor" in path
    is_outdoor = "Outdoor" in path

    stats["total"] += 1
    if is_indoor:
        if is_masked:
            stats["indoor_masked_total"] += 1
        else:
            stats["indoor_nomask_total"] += 1
    elif is_outdoor:
        if is_masked:
            stats["outdoor_masked_total"] += 1
        else:
            stats["outdoor_nomask_total"] += 1


    t_yolo_start = time.perf_counter()
    detections = run_face_inference_test(person_model, face_model, frame)
    t_yolo_end = time.perf_counter()
    yolo_ms = (t_yolo_end - t_yolo_start) * 1000

    if len(detections) == 0:
        print(f"{filename:<35} | {real_name:<10} | {'---':<15} | {'0.00'}  | -")
        stats["no_face_detected"] += 1
        continue

    best_match_name = "Unknown"
    best_match_score = 0.0
    was_upscaled = False

    for det in detections:
        box = det['face_box']


        res_emb, upscaled_flag, timings = get_embedding(frame, box, embed_model)

        if res_emb is None: continue


        print(
            f"   ⏱️ [Face]: YOLO={yolo_ms:.1f}ms | Prep={timings['prep_ms']:.1f}ms | SR={timings['sr_ms']:.1f}ms | Net={timings['infer_ms']:.1f}ms")

        vector = res_emb / np.linalg.norm(res_emb)

        local_best_name = "Unknown"
        local_max_sim = -1.0

        for db_name, db_vector in face_db.items():
            sim = np.dot(vector, db_vector)
            if sim > local_max_sim:
                local_max_sim = sim
                local_best_name = db_name

        if local_max_sim > best_match_score:
            best_match_score = local_max_sim
            was_upscaled = upscaled_flag
            if local_max_sim >= THRESHOLD:
                best_match_name = local_best_name
            else:
                best_match_name = "Unknown"

        if VISUALIZE:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            is_correct = (local_best_name == real_name and local_max_sim >= THRESHOLD)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            border_col = (0, 215, 255) if upscaled_flag else color
            cvzone.cornerRect(frame, [x1, y1, w, h], l=10, rt=1, colorR=border_col)


    status = ""
    if best_match_name == real_name:
        status = "✅"
        stats["correct"] += 1
        if is_indoor:
            if is_masked:
                stats["indoor_masked_correct"] += 1
            else:
                stats["indoor_nomask_correct"] += 1
        elif is_outdoor:
            if is_masked:
                stats["outdoor_masked_correct"] += 1
            else:
                stats["outdoor_nomask_correct"] += 1
    elif best_match_name == "Unknown":
        status = "❔"
        stats["unknown"] += 1
    else:
        status = "❌"
        stats["wrong"] += 1

    sr_tag = "⚡" if was_upscaled else " "
    print(f"{filename:<35} | {real_name:<10} | {best_match_name:<15} | {best_match_score:.2f}  | {status} | {sr_tag}")

    if VISUALIZE:
        info_text = f"{best_match_name} ({best_match_score:.2f}) {status}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        h_disp, w_disp = frame.shape[:2]
        if h_disp > 800:
            frame = cv2.resize(frame, (int(w_disp * 800 / h_disp), 800))
        cv2.imshow("Test MobileFaceNet + FSRCNN", frame)
        if cv2.waitKey(1) == ord('q'): break

if VISUALIZE:
    cv2.destroyAllWindows()



def calc_acc(correct, total):
    return (correct / total) * 100 if total > 0 else 0


acc_total = calc_acc(stats["correct"], stats["total"])

print("\n" + "=" * 55)
print("📊 RAPORT FINAL DETALIAT (MobileFaceNet + SR)")
print("=" * 55)
print(f"Total Imagini Procesate: {stats['total']}")
print(f" -> Fețe Nedetectate: {stats['no_face_detected']}")
print(f" -> Corecte Total:    {stats['correct']}")
print(f" -> Greșite Total:    {stats['wrong']}")
print(f" -> Necunoscute:      {stats['unknown']}")
print("-" * 55)
print(f"🎯 ACURATEȚE TOTALĂ: {acc_total:.2f}%")
print("-" * 55)
print("📂 STATISTICI PE CATEGORII (Corecte / Total):")
print(
    f"   🏠 Indoor  - Fără Mască: {stats['indoor_nomask_correct']}/{stats['indoor_nomask_total']}  ({calc_acc(stats['indoor_nomask_correct'], stats['indoor_nomask_total']):.2f}%)")
print(
    f"   🏠 Indoor  - Cu Mască:   {stats['indoor_masked_correct']}/{stats['indoor_masked_total']}  ({calc_acc(stats['indoor_masked_correct'], stats['indoor_masked_total']):.2f}%)")
print(
    f"   🌳 Outdoor - Fără Mască: {stats['outdoor_nomask_correct']}/{stats['outdoor_nomask_total']} ({calc_acc(stats['outdoor_nomask_correct'], stats['outdoor_nomask_total']):.2f}%)")
print(
    f"   🌳 Outdoor - Cu Mască:   {stats['outdoor_masked_correct']}/{stats['outdoor_masked_total']} ({calc_acc(stats['outdoor_masked_correct'], stats['outdoor_masked_total']):.2f}%)")
print("=" * 55)
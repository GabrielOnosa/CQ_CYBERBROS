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

from model import MobileFaceNet

#cale pt train
TRAIN_IMAGE_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\db\cristi\train.jpeg'

# calea pt test
TEST_IMAGE_PATH = r'C:\Users\raduc\Downloads\dataset\dataset\test\cristi\test.jpeg'

DB_FILE = 'face_db.pkl'
WEIGHTS_PATH = 'model_mobilefacenet.pth'

# Numele
NEW_PERSON_NAME = "Cristi"

# Praguri
CONF_PERSON = 0.25
CONF_FACE = 0.20
THRESHOLD_ID = 0.13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" System ready using: {device}")



def run_detection_pipeline(person_model, face_model, frame, debug_title="DEBUG"):
    h_img, w_img = frame.shape[:2]
    all_detections = []

    person_result = person_model.predict(frame, conf=CONF_PERSON, imgsz=1216, verbose=False)

    for info in person_result:
        for box in info.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # Person
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Padding
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.10)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w_img, x2 + pad_x)
                crop_y2 = min(h_img, y2 + pad_y)

                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if person_crop.size > 0:
                    # Upscale
                    h_c, w_c = person_crop.shape[:2]
                    scale_factor = 2
                    try:
                        person_crop_upscaled = cv2.resize(person_crop, (w_c * scale_factor, h_c * scale_factor))
                    except:
                        continue


                    face_results = face_model.predict(person_crop_upscaled, conf=CONF_FACE, imgsz=640, verbose=False)

                    for f_info in face_results:
                        for f_box in f_info.boxes:
                            fx1, fy1, fx2, fy2 = f_box.xyxy[0]


                            fx1_real = fx1 / scale_factor
                            fy1_real = fy1 / scale_factor
                            fx2_real = fx2 / scale_factor
                            fy2_real = fy2 / scale_factor

                            global_fx1 = int(crop_x1 + fx1_real)
                            global_fy1 = int(crop_y1 + fy1_real)
                            global_fx2 = int(crop_x1 + fx2_real)
                            global_fy2 = int(crop_y1 + fy2_real)

                            all_detections.append({
                                "face_box": [global_fx1, global_fy1, global_fx2, global_fy2]
                            })


                            cvzone.cornerRect(frame, [global_fx1, global_fy1, global_fx2 - global_fx1,
                                                      global_fy2 - global_fy1], l=10, rt=1, colorR=(255, 0, 0))

    return all_detections



preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(full_image_bgr, coords, model):
    x1, y1, x2, y2 = coords
    h, w = full_image_bgr.shape[:2]

    # Padding mic pentru embedding
    pad_w = int((x2 - x1) * 0.1)
    pad_h = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    if x2 - x1 < 10 or y2 - y1 < 10: return None

    face_crop = full_image_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)


    t1 = preprocess(face_rgb).unsqueeze(0).to(device)


    with torch.no_grad():
        embedding = model(t1).cpu().numpy()[0]


    return embedding



print("loading models")
person_model = YOLO('yolo12s.pt')
face_model = YOLO('yolov12m-face.pt')
embed_model = MobileFaceNet(512).to(device)

try:
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if 'state_dict' in checkpoint:
        embed_model.load_state_dict(checkpoint['state_dict'])
    else:
        embed_model.load_state_dict(checkpoint)
    embed_model.eval()
except Exception as e:
    print(f"Eroare MobileFaceNet: {e}")
    sys.exit()

# --- INCARCARE DB EXISTENT ---
if os.path.exists(DB_FILE):
    with open(DB_FILE, 'rb') as f:
        face_db = pickle.load(f)
    print(f"Baza de date incarcata: {len(face_db)} persoane.")
else:
    face_db = {}
    print("Baza de date goala/inexistenta.")


print(f"\n INVATARE: Procesez imaginea de TRAIN: {TRAIN_IMAGE_PATH}")
train_img = cv2.imread(TRAIN_IMAGE_PATH)

if train_img is None:
    print(f"nu pot citi imaginea de train: {TRAIN_IMAGE_PATH}")
    sys.exit()


train_detections = run_detection_pipeline(person_model, face_model, train_img, "DEBUG TRAIN")

if len(train_detections) == 0:
    print("Nu am gasit nicio fata in imaginea de TRAIN!")
    sys.exit()


best_train_face = max(train_detections,
                      key=lambda d: (d['face_box'][2] - d['face_box'][0]) * (d['face_box'][3] - d['face_box'][1]))
train_vec = get_embedding(train_img, best_train_face['face_box'], embed_model)

if train_vec is not None:

    train_vec = train_vec / np.linalg.norm(train_vec)


    face_db[NEW_PERSON_NAME] = train_vec
    print(f"{NEW_PERSON_NAME} a fost adaugat cu succes in baza de date (memorie)!")


else:
    print("Eroare la generarea vectorului pentru TRAIN.")
    sys.exit()


print(f"\n TESTARE: Procesez imaginea de TEST: {TEST_IMAGE_PATH}")
test_img = cv2.imread(TEST_IMAGE_PATH)

if test_img is None:
    print("Nu pot citi imaginea de test!")
    sys.exit()


test_detections = run_detection_pipeline(person_model, face_model, test_img, "DEBUG TEST")

print(f"   Am detectat {len(test_detections)} fete in imaginea de test.")


for i, det in enumerate(test_detections):
    box = det['face_box']
    current_vector = get_embedding(test_img, box, embed_model)

    if current_vector is None: continue
    current_vector = current_vector / np.linalg.norm(current_vector)

    best_name = "Unknown"
    max_similarity = -1.0

    for db_name, db_vector in face_db.items():
        score = np.dot(current_vector, db_vector)
        if score > max_similarity:
            max_similarity = score
            best_name = db_name


    if max_similarity >= THRESHOLD_ID:
        status_text = f"FOUND: {best_name} ({max_similarity:.2f})"
        color = (0, 255, 0)
    else:
        status_text = f"Unknown ({best_name}: {max_similarity:.2f})"
        color = (0, 0, 255)

    print(f"   Fața #{i + 1}: {status_text}")


    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cvzone.cornerRect(test_img, [x1, y1, w, h], l=15, rt=1, colorR=color)
    cvzone.putTextRect(test_img, f"{best_name} {max_similarity:.2f}", (max(0, x1), max(20, y1 - 10)), scale=1,
                       thickness=1, colorR=color)


h, w = test_img.shape[:2]
if h > 900: test_img = cv2.resize(test_img, (int(w * 900 / h), 900))

cv2.imshow("TEST REZULTAT - CRISTI", test_img)
print("\nApasa orice tasta pentru a inchide...")
cv2.waitKey(0)
cv2.destroyAllWindows()
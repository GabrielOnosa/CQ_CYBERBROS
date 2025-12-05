import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import MobileFaceNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rulez pe: {device}")

# YOLO
try:

    facemodel = YOLO('yolov12m-face.pt')
except Exception as e:
    print(f"EROARE YOLO: {e}")
    exit()


embedding_size = 512
rec_model = MobileFaceNet(embedding_size).to(device)
weights_path = 'model_mobilefacenet.pth'

try:
    checkpoint = torch.load(weights_path, map_location=device)


    if 'state_dict' in checkpoint:
        rec_model.load_state_dict(checkpoint['state_dict'])
    else:
        rec_model.load_state_dict(checkpoint)

    rec_model.eval()
    print("✅ MobileFaceNet încărcat cu succes!")
except Exception as e:
    print(f"❌ EROARE MobileFaceNet: {e}")
    print("Verifică dacă fișierul .pth este corect sau încearcă embedding_size=128")
    exit()


preprocess = transforms.Compose([
    transforms.Resize((112, 112)),  # MobileFaceNet =>  112x112
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


im_path = r'C:\Users\raduc\Downloads\dataset\dataset\test\Outdoor\Non-masked\Pablo - Outdoor - 9C.png'
frame = cv2.imread(im_path)

if frame is None:
    print("EROARE: Nu am putut citi imaginea. Verifică calea!")
    exit()

h_img, w_img = frame.shape[:2]

# 3. Predict YOLO
# imgsz=1216 (multiplu de 32)
face_result = facemodel.predict(frame, conf=0.4, imgsz=1216, verbose=False)

# 4. etragere fete, calcul embedding
for info in face_result:
    parameters = info.boxes
    for box in parameters:

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        #safe crop
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        w, h = x2 - x1, y2 - y1

        # fata prea mica, o ignor
        if w < 10 or h < 10:
            continue

        #decupare ,preprocesoare mobilefacenet
        face_crop = frame[y1:y2, x1:x2]  # decupare (bgr)

        # coonvert bgr -> rgb
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        # transform in tensor si batch dimension (1, 3, 112, 112)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        #  inferenta
        with torch.no_grad():
            embedding = rec_model(input_tensor)

            feature_vector = embedding.cpu().numpy()[0]


        print(f"Fața detectată la [{x1}, {y1}] -> Vector generat (dimensiune {feature_vector.shape})")
        print(feature_vector)

        # desenare
        cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3, colorR=(0, 255, 0))
        cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 5. Afisare
scale = 1080 / h_img
dim = (int(w_img * scale), 1080)
display_frame = cv2.resize(frame, dim)

cv2.imshow('Rezultat Detectie + Embeddings', display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
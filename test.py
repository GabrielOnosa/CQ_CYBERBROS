import torch
import ultralytics
from ultralytics import YOLO

# încarcă un model pre-antrenat
model = YOLO("yolov8n.pt")

# rulează o predicție pe o imagine de test
results = model.predict(source="https://ultralytics.com/images/bus.jpg", show=True)
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import cvzone
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
<<<<<<< HEAD
from model import MobileFaceNet
=======
from Models import MobileFaceNet
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0

class customImageDataset(Dataset):
    def __init__(self, directory, transform = None):
        self.directories_list = []
        self.directory = directory
        self.transform = transform
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            for image in os.listdir(folder_path):
                self.directories_list.append(os.path.join(folder_path,image))
        #print(self.directories_list)

    def __len__(self):
        return len(self.directories_list)

    def __getitem__(self, idx):
        directory =  self.directories_list[idx]
        image_array = Image.open(directory).convert('RGB')
        image_array = np.array(image_array)
        if self.transform is not None:
            image_array = self.transform(image_array)
        return image_array

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((112, 112))
                                            ])


def run_face_inference_train(model, frame):
    # 1. Run prediction
    # model.predict returns a list of Result objects (one per frame)
    results = model.predict(frame, verbose=False)

    # 2. Extract the result for the first (and only) frame
    result = results[0]

    # 3. Check if any detections exist
    if len(result.boxes) == 0:
        # Return specific values indicating failure (e.g., None or 0s)
        return None, None, None, None

    # 4. Sort and pick the best one
    # We use python's 'sorted' on result.boxes
    # x.conf[0] gets the float value from the tensor
    sorted_boxes = sorted(result.boxes, key=lambda x: x.conf[0], reverse=True)

    # 5. Take the first element (Best Confidence)
    best_box = sorted_boxes[0]

    # 6. Extract Coordinates
    x1, y1, x2, y2 = best_box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return x1, y1, x2, y2



def run_face_inference_test(person_model, face_model, frame):

<<<<<<< HEAD

    h_img, w_img = frame.shape[:2]

    all_detections = []


=======
    # 1. Get Image Dimensions (Needed for boundary checks)
    h_img, w_img = frame.shape[:2]

    # 2. Container for all found faces
    all_detections = []

    # 3. Person Detection
    # Use the passed argument 'person_model', not 'classic_model'
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
    person_result = person_model.predict(frame, conf=0.4, imgsz=1200, verbose=False)

    for info in person_result:
        parameters = info.boxes
        for box in parameters:
            cls_id = int(box.cls[0])

            if cls_id == 0:  # If Person
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # --- STEP A: SAFE PADDING ---
                pad_x = int(w * 0.5)
                pad_y = int(h * 0.1)

                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w_img, x2 + pad_x)
                crop_y2 = min(h_img, y2 + pad_y)

                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

<<<<<<< HEAD

=======
                # --- STEP B: UPSCALE & DETECT ---
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                if person_crop.size > 0:
                    scale_factor = 3
                    h_c, w_c = person_crop.shape[:2]

<<<<<<< HEAD
=======
                    # Upscale
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                    person_crop_upscaled = cv2.resize(person_crop,
                                                      (w_c * scale_factor, h_c * scale_factor),
                                                      interpolation=cv2.INTER_CUBIC)

<<<<<<< HEAD

=======
                    # Predict Face
                    # Use passed argument 'face_model'
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                    face_results = face_model.predict(person_crop_upscaled, conf=0.4, imgsz=640, verbose=False)

                    for f_info in face_results:
                        for f_box in f_info.boxes:
                            fx1, fy1, fx2, fy2 = f_box.xyxy[0]

<<<<<<< HEAD

=======
                            # --- STEP C: COORDINATE MAPPING ---
                            # Map back to original scale
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                            fx1_real = fx1 / scale_factor
                            fy1_real = fy1 / scale_factor
                            fx2_real = fx2 / scale_factor
                            fy2_real = fy2 / scale_factor

<<<<<<< HEAD

=======
                            # Map to global image
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                            global_fx1 = int(crop_x1 + fx1_real)
                            global_fy1 = int(crop_y1 + fy1_real)
                            global_fx2 = int(crop_x1 + fx2_real)
                            global_fy2 = int(crop_y1 + fy2_real)

<<<<<<< HEAD

=======
                            # Add to results list
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
                            all_detections.append({
                                "face_box": [global_fx1, global_fy1, global_fx2, global_fy2]
                            })

    return all_detections

if __name__ == '__main__':
<<<<<<< HEAD
    face_model = YOLO('yolov12m-face.pt')
    person_model = YOLO('yolo11n.pt')

    people_dataset = customImageDataset(r'C:\Users\raduc\Downloads\dataset\dataset\db')
=======
    face_model = YOLO('yolov12l-face.pt')
    person_model = YOLO('yolo11n.pt')

    people_dataset = customImageDataset(r'D:\Downloads\dataset_hackathon\dataset\db')
>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0
    coord_list = (run_face_inference_test(person_model, face_model, people_dataset[1]))
    img_path = people_dataset[1]
    frame = np.array(img_path)
    frame2 = np.array(img_path)
    for i in range(len(coord_list)):
        global_fx1 = coord_list[i]['face_box'][0]
        global_fy1 = coord_list[i]['face_box'][1]

        fw = coord_list[i]['face_box'][2] - coord_list[i]['face_box'][0]
        fh = coord_list[i]['face_box'][3] - coord_list[i]['face_box'][1]

        cvzone.cornerRect(frame, [global_fx1, global_fy1, fw, fh], l=5, rt=2, colorR=(255, 0, 255))
    plt.imshow(frame)
    plt.show()

    coord_list2 = run_face_inference_train(face_model, frame2)
    x1 = coord_list2[0]
    y1 = coord_list2[1]
    x2 = coord_list2[2]
    y2 = coord_list2[3]
    w = x2-x1
    h = y2-y1
    cvzone.cornerRect(frame2, [x1, y1, w, h], l=5, rt=2, colorR=(255, 0, 255))
    plt.imshow(frame2)
    plt.show()
<<<<<<< HEAD
=======
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Rulez pe: {device}")
# #pipeline for extracting face
#
# # Load Models
# facemodel = YOLO('yolov12l-face.pt')
# classic_model = YOLO('yolo11n.pt')
#
# # Load Image
# im_path = r'D:\Downloads\dataset_hackathon\dataset\test\Outdoor\Masked\Andres - Outdoor - M30C.png'
# image_array = Image.open(im_path).convert('RGB')
# frame = np.array(image_array)
#
# h_img, w_img, _ = frame.shape
#
# # 1. Detect Persons
# # Reduced conf to 0.25 to ensure we catch people even if blurry


>>>>>>> ab710c3604a1cc5f983d34a69cf0c7198714f5f0

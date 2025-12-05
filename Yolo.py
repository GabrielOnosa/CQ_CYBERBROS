import cvzone
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#pipeline for extracting face

# Load Models
facemodel = YOLO('yolov12l-face.pt')
classic_model = YOLO('yolo11n.pt')

# Load Image
im_path = r'D:\Downloads\dataset_hackathon\dataset\test\Outdoor\Masked\Cristina - Outdoor - M30C.png'
image_array = Image.open(im_path).convert('RGB')
frame = np.array(image_array)

h_img, w_img, _ = frame.shape

# 1. Detect Persons
# Reduced conf to 0.25 to ensure we catch people even if blurry
face_result = classic_model.predict(frame, conf=0.4, imgsz=1200)

for info in face_result:
    parameters = info.boxes
    for box in parameters:
        cls_id = int(box.cls[0])

        if cls_id == 0:  # If Person
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # --- STEP A: SAFE PADDING LOGIC ---
            # You wanted to expand the box to get context.
            # Let's add 50% width padding on both sides.
            pad_x = int(w * 0.5)
            pad_y = int(h * 0.1)  # Add a little vertical padding too

            # Calculate new crop coordinates with boundary checks (max/min)
            # This prevents the "Negative Index" crash
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w_img, x2 + pad_x)
            crop_y2 = min(h_img, y2 + pad_y)

            # Perform the Crop
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # --- STEP B: UPSCALE & DETECT ---
            if person_crop.size > 0:
                scale_factor = 3
                h_c, w_c = person_crop.shape[:2]

                # Upscale
                person_crop_upscaled = cv2.resize(person_crop,
                                                  (w_c * scale_factor, h_c * scale_factor),
                                                  interpolation=cv2.INTER_CUBIC)
                plt.imshow(person_crop_upscaled)
                plt.show()

                # Predict Face (Low confidence for blurry faces)
                # imgsz=640 is usually better than 320 for small object detection
                face_results = facemodel.predict(person_crop_upscaled, conf=0.2, imgsz=640, verbose=True)

                for f_info in face_results:
                    for f_box in f_info.boxes:
                        fx1, fy1, fx2, fy2 = f_box.xyxy[0]

                        # --- STEP C: THE CORRECT MATH ---

                        # 1. Downscale: Map coordinates back to the original crop size
                        # We must divide by the scale factor
                        fx1_real = fx1 / scale_factor
                        fy1_real = fy1 / scale_factor
                        fx2_real = fx2 / scale_factor
                        fy2_real = fy2 / scale_factor

                        # 2. Global Offset: Map relative crop coords to full 4K image
                        # CRITICAL: Use 'crop_x1', NOT 'x1'.
                        # 'x1' is the person box. 'crop_x1' is the padded image start.
                        global_fx1 = int(crop_x1 + fx1_real)
                        global_fy1 = int(crop_y1 + fy1_real)
                        global_fx2 = int(crop_x1 + fx2_real)
                        global_fy2 = int(crop_y1 + fy2_real)

                        # Width/Height
                        fw = global_fx2 - global_fx1
                        fh = global_fy2 - global_fy1

                        # Draw Face
                        cvzone.cornerRect(frame, [global_fx1, global_fy1, fw, fh],
                                          l=5, rt=2, colorR=(255, 0, 255))

            # Draw Person Box (Optional)
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3, colorR=(0, 255, 0))

# Display Logic
display_frame = cv2.resize(frame, (1920, 1080))
cv2.imshow('frame', display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


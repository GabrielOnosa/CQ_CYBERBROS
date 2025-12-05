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
from model import MobileFaceNet
from Yolo import customImageDataset
from Yolo import run_face_inference_test
from Yolo import run_face_inference_train

dataset = customImageDataset(r'D:\Downloads\dataset_hackathon\dataset\db')
#training loop type shit
model = YOLO('yolov12m-face.pt')
for i in range(len(dataset)):
    im_path = dataset[i]

    cropped_image_coordinates = run_face_inference_train(model, dataset[i])

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((112, 112))])
'''
import ultralytics
import torch
import cv2
from ultralytics import YOLO

# Checking whether it's running on GPU
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Printing YOLO version
print(ultralytics.__version__)

# Load YOLOv8 segmentation model (custom trained)
model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train3\weights\best.pt")
print(model.names)

# Open the video file
video_path = r"D:\learn\cmputer_Vision_opencv\videoplayback_converted.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the frame (adjust conf and classes as per your need)
    model.predict(frame, show=True, conf=0.10)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
'''

import ultralytics
import torch
import cv2
import time
from ultralytics import YOLO

# Check for GPU
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Print YOLO version
print(ultralytics.__version__)

# Load YOLOv8 segmentation model (custom trained)
#model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\Final\weights\best.pt")
# model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train3\weights\best.pt")
# model = YOLO(r"D:\learn\cmputer_Vision_opencv\best.pt")
model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train4_batch70%\weights\best.pt") #Working
# model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train2_epochs=1000,dataset\weights\custom_best.pt") #will not work 
# model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train5_using_prev_train4\weights\best.pt")
# model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train\weights\best.pt")


print(model.names)

# Open video
video_path = r"D:\learn\cmputer_Vision_opencv\videos\Yellowstone Wolves...an unforgettable encounter....mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# (Optional) Frame Skipping toggle — use True to skip every 2 out of 3 frames
enable_skip = False
frame_count = 0

# Inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if enable_skip and frame_count % 3 != 0:
        continue

    # Resize for speed (smaller = faster, adjust if needed)
    frame = cv2.resize(frame, (640, 700))

    # Run inference with streaming
    results = model.predict(frame, conf=0.1, stream=True)

    # Plot and show result
    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLOv8 Segmentation", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

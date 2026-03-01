import cv2
import os
import subprocess
from ultralytics import YOLO
import torch

# ✅ Input: single video path (original)
video_path = r"D:\learn\cmputer_Vision_opencv\videos\Wonderful Wildlife Encounter when Rhino and Zebra share a waterhole at Kruger National Park (online-video-cutter.com).mp4"

# ✅ Check for GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Running on CPU")

# ✅ Load YOLO model
model = YOLO(r"D:\learn\cmputer_Vision_opencv\runs\segment\train_yaml_epoch1000_batch0p7\weights\best.pt") #Working
  # Change to your .pt model if needed
print("Loaded model:", model.names)

# ✅ Try to open video
cap = cv2.VideoCapture(video_path)

# ❌ If it fails, try to convert using FFmpeg
if not cap.isOpened():
    print("⚠️ Cannot open video. Trying conversion with FFmpeg...")

    # Generate new converted path with "_converted" suffix
    base, ext = os.path.splitext(video_path)
    converted_path = base + "_converted" + ext

    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", "-strict", "experimental",
        converted_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        cap = cv2.VideoCapture(converted_path)
        print(f"✅ Converted and using: {converted_path}")
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg or add to PATH.")
        exit()

# ❌ Still fails?
if not cap.isOpened():
    print("❌ Failed to open video after conversion.")
    exit()

# ✅ Inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model.predict(frame, conf=0.5, stream=True)

    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLOv8 Segmentation", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

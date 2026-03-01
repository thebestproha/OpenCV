import ultralytics
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

#checking whether its running on gpu
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

#printing yolo version
print(ultralytics.__version__)

# Load YOLOv8 segmentation model (you can use 'yolov8s-seg.pt', 'yolov8m-seg.pt', etc.)
base_dir = Path(__file__).resolve().parent
model = YOLO(str(base_dir / "runs" / "segment" / "train_yaml_epoch1000_batch16" / "weights" / "best.pt"))
print(model.names)
# Read the image
image_path = str(base_dir / "Images" / "tiger.jpg") # Change this to your image path
image = cv2.imread(image_path)
#model.predict(image,show=True,conf=0.70,classes=[2])   , class for only cars
model.predict(image,show=True,conf=0.10)   #config for 0.7 will show only if 70 percentage is sure

cv2.waitKey(0)
cv2.destroyAllWindows()

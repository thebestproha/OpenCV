import torch
torch.cuda.set_device(0)
import cv2
from ultralytics import YOLO
import ultralytics
def train_model():                          #checking if runnning on gpu
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    print(ultralytics.__version__) #prinitng version of yolo
    # Load YOLOv8 segmentation model (you can use 'yolov8s-seg.pt', 'yolov8m-seg.pt', etc.)

    model = YOLO("yolov8s-seg.yaml")
   # model = YOLO(rf"D:\learn\cmputer_Vision_opencv\runs\segment\train3\weights\best.pt")
    image_path = rf"D:\learn\Animal Project.v3i.yolov8\data.yaml"
    results = model.train(data=image_path, epochs=1000, batch=0.70,imgsz=640, device=0)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, only for frozen apps
    train_model()

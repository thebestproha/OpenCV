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

    #model = YOLO("yolov8s-seg.yaml")
    model = YOLO(rf"D:\learn\cmputer_Vision_opencv\runs\segment\train_yaml_epoch1000_batch0p7\weights\best.pt")
    image_path = rf"D:\learn\Dataset for animals\data.yaml"
    results = model.train(data=image_path, epochs=1000, batch=0.80,imgsz=640, device=0) #batch 80% ram usage , integer = no of images

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, only for frozen apps
    train_model()

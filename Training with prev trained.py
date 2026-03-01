import torch
torch.cuda.set_device(0)
import cv2
from pathlib import Path
from ultralytics import YOLO
import ultralytics
def train_model():                          #checking if runnning on gpu
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    print(ultralytics.__version__) #prinitng version of yolo
    # Load YOLOv8 segmentation model (you can use 'yolov8s-seg.pt', 'yolov8m-seg.pt', etc.)

    base_dir = Path(__file__).resolve().parent

    #model = YOLO("yolov8s-seg.yaml")
    model = YOLO(str(base_dir / "runs" / "segment" / "train_yaml_epoch1000_batch0p7" / "weights" / "best.pt"))
    image_path = str(base_dir / "datasets" / "Dataset for animals" / "data.yaml")
    results = model.train(data=image_path, epochs=1000, batch=0.80,imgsz=640, device=0) #batch 80% ram usage , integer = no of images

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, only for frozen apps
    train_model()

# Animal/Wildlife Monitoring with YOLOv8 Segmentation

This project is a practical wildlife monitoring pipeline built with YOLOv8 segmentation.
The dataset went through a multi-stage annotation and validation process in Roboflow (AI-assisted + manual high-detail correction), then moved into repeated local experiments with multiple training and inference methods.

1. Roboflow training/testing inside the platform.
2. Local training/testing using Python + Ultralytics + OpenCV.

The current class set is:
`Rhinoceroses`, `Tiger`, `Wolf`, `Wolf_cub`, `Zebra`.

## Quick beginner pipeline (project overview)

If you are new to CV projects, this repository follows this simple flow:

1. **Get data** → collect raw wildlife images/videos.
2. **Annotate & prepare** → start with AI-assisted labels, then manually correct masks/classes.
3. **Split dataset** → prepare train/validation/test sets (commonly 80/20 or train/val/test splits).
4. **Train model** → run segmentation training and compare multiple experiments.
5. **Validate/Test** → check mAP and other metrics, then choose best checkpoint.
6. **Predict on new data** → run image/video inference using the selected weights.
7. **Iterate** → improve labels, retrain, and compare runs again.

In this project specifically, exported datasets are organized as `train`, `valid`, and `test` folders and are consumed by YOLO through `data.yaml`.

## 0) Initial workflow (before local code)

1. Collect raw wildlife images/videos from your own sources and organize them by scene/species.
2. Upload the raw data to Roboflow project `animal-project-tlzox`.
3. Run Roboflow AI annotation to generate initial labels.
4. Manually refine and correct annotations class-by-class (high-detail masks/labels) and finish class assignment.
5. Generate dataset versions in Roboflow (v2 and later v3 in this project).
6. Train/test inside Roboflow first to validate label quality, class balance, and baseline performance.
7. Export the validated dataset (YOLOv8 format) and use that export directly in local code for image/video prediction.

This keeps the workflow clear: data preparation and annotation quality checks happen first in Roboflow, then reproducible model experiments are run locally.

## Workflow summary across versions/methods

- **Method A (Roboflow platform path):** AI annotation → manual correction/classification → Roboflow train/test → validated dataset export.
- **Method B (Local code path):** Use exported dataset YAML in Ultralytics scripts for repeated training runs (`train_pretrained_pt_epoch10`, `train_yaml_epoch10_failed`, `train_yaml_epoch1000_batch16`, `train_yaml_epoch1000_batch0p7`, `train_from_train_yaml_epoch1000_batch0p7`, `train_final_yaml_epoch1000_batch0p7_v3`) and compare metrics.
- **Prediction path:** Use trained checkpoints in image and video scripts to run direct class prediction on custom wildlife photos and videos.

This project intentionally uses and compares two training/testing methods:
1. Roboflow inbuilt training/testing pipeline.
2. Custom local training/testing pipeline using your own Python code.

## 1) Dataset and annotation source

- Roboflow workspace: `computervision-tmgvz`
- Roboflow project: `animal-project-tlzox`
- Dataset versions used in experiments: v2 and v3 (YOLOv8 export format)
- License noted in source exports: CC BY 4.0

Both exports use train/val/test splits and the same 5 classes.

## 2) Project structure (important folders)

This repository now includes code, trained runs, model weights, and datasets so it can be recreated directly after cloning.

```text
.
├─ Training.py
├─ Training with prev trained.py
├─ custom_dataset___Detection.py
├─ demodetenction.py
├─ video_detector.py
├─ Video_default.py
├─ new_video_custom.py
├─ runs/segment/
│  ├─ train_pretrained_pt_epoch10/
│  ├─ train_yaml_epoch10_failed/
│  ├─ train_yaml_epoch1000_batch16/
│  ├─ train_yaml_epoch1000_batch0p7/
│  ├─ train_from_train_yaml_epoch1000_batch0p7/
│  └─ train_final_yaml_epoch1000_batch0p7_v3/
├─ datasets/
│  ├─ Animal Project.v3i.yolov8/
│  ├─ Dataset for animals/
│  └─ animals training images/
├─ Images/
└─ videos/
```

## 3) What each code file does (and why)

- [Training.py](Training.py#L1-L23)
	- Trains segmentation from YOLO config (`yolov8s-seg.yaml`) for long runs (`epochs=1000`) on dataset config.
	- Use this when you want a fresh experiment from model config.

- [Training with prev trained.py](Training%20with%20prev%20trained.py#L1-L23)
	- Starts from a previously trained checkpoint (`train_yaml_epoch1000_batch0p7/weights/best.pt`) and continues training.
	- Use this for fine-tuning or continuation from a strong earlier run.

- [custom_dataset___Detection.py](custom_dataset___Detection.py#L1-L25)
	- Runs image inference using a custom trained checkpoint.
	- Use this for quick quality checks on single photos.

- [demodetenction.py](demodetenction.py#L1-L25)
	- Runs image inference using base `yolov8s-seg.pt`.
	- Use this as a baseline comparison against custom-trained weights.

- [video_detector.py](video_detector.py#L47-L114)
	- Video inference with custom checkpoint, optional frame skipping, and real-time display.
	- Use this when testing model behavior on longer wildlife clips.

- [Video_default.py](Video_default.py#L1-L69)
	- Video inference with FFmpeg fallback conversion if OpenCV cannot open source video.
	- Use this for robust playback across tricky codecs.

- [new_video_custom.py](new_video_custom.py#L1-L70)
	- Similar to `Video_default.py` but already pointed to custom trained weights.
	- Use this as the custom-model video demo path.

## 4) Training runs: differences and evaluation

The following numbers are read directly from each run’s `results.csv` and `args.yaml`.

| Run folder | Setup difference | Best checkpoint summary (Mask metrics) |
|---|---|---|
| `train_pretrained_pt_epoch10` | `model=yolov8s-seg.pt`, `epochs=10`, `batch=16`, dataset v2 | best epoch 10, mAP50-95(M)=0.64296, mAP50(M)=0.88085 |
| `train_yaml_epoch10_failed` | `model=yolov8s-seg.yaml`, `epochs=10`, `batch=16`, dataset v2 | best epoch 9, mAP50-95(M)=0.00318, mAP50(M)=0.00939 (weak run; kept for comparison/history) |
| `train_yaml_epoch1000_batch16` | `model=yolov8s-seg.yaml`, `epochs=1000`, `batch=16`, dataset v2 | best epoch 409, mAP50-95(M)=0.64839, mAP50(M)=0.85690 |
| `train_yaml_epoch1000_batch0p7` | `model=yolov8s-seg.yaml`, `epochs=1000`, `batch=0.7`, dataset v2 | best epoch 415, mAP50-95(M)=0.62594, mAP50(M)=0.81316 |
| `train_from_train_yaml_epoch1000_batch0p7` | starts from `train_yaml_epoch1000_batch0p7/weights/best.pt`, `epochs=1000`, `batch=0.8`, dataset v2 | best epoch 1, mAP50-95(M)=0.62140, mAP50(M)=0.82281 |
| `train_final_yaml_epoch1000_batch0p7_v3` | `model=yolov8s-seg.yaml`, `epochs=1000`, `batch=0.7`, dataset v3 | best epoch 485, mAP50-95(M)=0.62845, mAP50(M)=0.79642 |

Run logs are available locally under `runs/segment/*/results.csv` and `runs/segment/*/args.yaml`.

## 5) Run naming convention used

Run folders are now named directly by training setup, so labels are built into the folder names:

- `train_pretrained_pt_epoch10`
- `train_yaml_epoch10_failed`
- `train_yaml_epoch1000_batch16`
- `train_yaml_epoch1000_batch0p7`
- `train_from_train_yaml_epoch1000_batch0p7`
- `train_final_yaml_epoch1000_batch0p7_v3`

## 6) How to run locally

Install dependencies:

```bash
pip install ultralytics torch opencv-python
```

Train from config:

```bash
python Training.py
```

Continue from checkpoint:

```bash
python "Training with prev trained.py"
```

Image inference:

```bash
python custom_dataset___Detection.py
```

Video inference:

```bash
python video_detector.py
```

## 7) Reusing this pipeline for other CV domains

This same workflow can be reused to build new CV projects in different fields:

- **Medical imaging** (lesion/tumor/tool segmentation)
- **Industrial monitoring** (defect detection, equipment state monitoring)
- **Safety and surveillance** (PPE checks, intrusion zones, hazard detection)
- **Environmental monitoring** (species tracking, crop/forest observation)

What usually changes between domains:

1. Data source and quality requirements
2. Class definitions and annotation guidelines
3. Evaluation targets (for example higher recall for safety-critical tasks)
4. Deployment constraints (latency, hardware, camera type)

What stays the same:

- Data collection and annotation workflow
- Dataset versioning and train/valid/test split strategy
- Iterative experiment tracking across multiple runs
- Selecting best checkpoints and validating on new images/videos

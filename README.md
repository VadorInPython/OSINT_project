# Maritime Vessel Detection using OSINT and Computer Vision

## Project Overview
This project introduces the Danish Maritime Dataset (DMD), a new publicly available benchmark for maritime object detection in high-traffic coastal environments. The system leverages publicly available camera feeds from Danish fixed infrastructure to monitor the Great Belt Strait, one of Northern Europe's busiest international shipping corridors. This project is developed as part of a Master of Science in Autonomous Systems thesis research at the Technical University of Denmark (DTU).

## Objectives
* Release a high-quality, manually annotated maritime image dataset (DMD) to the public to support research on fixed-infrastructure monitoring.
* Provide baseline object detection benchmark results using YOLOv8 to establish reference performance for future comparative studies.
* Address real-world coastal surveillance challenges, including extreme scale variations, varying weather, low-light transitions, and specular sun glare.

## Methodology

### 1. Data Acquisition & Sensors
* Data is collected from two stationary vantage points: Camera East located on the east pylon of the Great Belt Bridge (providing a long-range, oblique, high-altitude view) and a camera on the island of Sprogø (providing a sea-level, horizontal view).
* The continuous camera feeds produce 1280x720 RGB JPG images.

### 2. Data Management
* All images are systematically annotated by hand with bounding boxes identifying visible maritime vessels.
* The dataset includes a temporally overlapping subset where both cameras simultaneously observe the waterway, facilitating multi-view research.

### 3. AI Model Development
* The system utilizes the YOLOv8n (nano) one-stage detector implemented via the Ultralytics library to balance computational efficiency with detection accuracy.
* The model was pretrained on existing large-scale datasets, specifically the Singapore Maritime Dataset (SMD) and SeaShips.
* Which data augmentation strategies were applied, including HSV photometric distortions, geometric transformations (random rotations and scaling), and mosaic augmentation.

## Project Structure
```text
├── .gitignore                                  # Git ignore file
├── README.md                                   # Project documentation
├── Storebaelt_cameras_acquistions.ipynb        # Data collection from webcams mounted at the Great Belt Bridge
├── demo.ipynb                                  # Jupyter notebook with model demonstrations
├── livestream_tracker.py                       # Python script for real-time vessel tracking
├── notebook_draft.ipynb                        # Experimental and draft analysis notebook
└── yolov8n.pt                                  # Pretrained YOLOv8 nano model weights

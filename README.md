# Fisheye YOLO Dataset Processor

## 📌 Introduction
This repository provides a script to transform a YOLO dataset from **Roboflow format** into a **fisheye dataset** by applying distortion transformations to both the images and their corresponding bounding boxes. This is useful when training object detection models on images captured by **fisheye cameras**, such as the **Intel RealSense T265**.

## ⚙️ Prerequisites
Before running the dataset processing script, ensure you have the following dependencies installed:

### 🛠 Required Libraries
- **Python 3.x**
- **OpenCV** (`cv2`) for image processing
- **NumPy** for numerical operations

To install the required dependencies, run:
```bash
pip install opencv-python numpy
```

### 📂 Dataset Format
This script is designed to work with a **YOLO dataset structured in Roboflow format**. The dataset should follow this directory structure:

```
Dataset/
│── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── labels/
│   │   ├── image1.txt
│   │   ├── image2.txt
│
│── valid/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── labels/
│   │   ├── image1.txt
│   │   ├── image2.txt
│
│── test/
│   ├── images/
│   ├── labels/
│
└── data.yaml
```

### 📏 Image Requirements
- **All images must have the same dimensions** before applying the fisheye transformation.
- The labels should be in **YOLO format** (i.e., each `.txt` file must contain `[class_id, x_center, y_center, width, height]`).


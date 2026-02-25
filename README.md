# Arrhythmia Detection Model

A lightweight ECG-based arrhythmia detection model built with a CNN architecture, trained in PyTorch, and optimized (via quantization and ONNX export) to run on an **Arduino Nano 33 BLE** — a microcontroller with only 1 MB of flash and 256 KB of RAM.

---

## Overview

This project tackles the challenge of running real-time cardiac arrhythmia classification directly on an edge device. The pipeline covers everything from ECG signal preprocessing to model training, export, and embedded deployment.

The model classifies ECG signals into arrhythmia categories and is compressed to fit within the strict memory constraints of the Arduino Nano 33 BLE using model quantization.

---

## Repository Structure

```
Arrhythmia-detection-model/
├── main.py               # Training and evaluation entry point
├── model.py              # CNN model architecture definition
├── preprocessing.py      # ECG signal preprocessing utilities
├── requirements.txt      # Python dependencies
├── best_ecg_model.pth    # Best trained PyTorch model weights
├── ecg_model.onnx        # Exported ONNX model (for TFLite/C conversion)
└── output_dir/           # Training outputs, logs, and exported artifacts
```

---

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for 1D ECG time-series classification. It is intentionally kept compact to enable deployment on microcontrollers after quantization.

Key design decisions:
- **1D convolutions** for temporal ECG feature extraction
- **Minimal parameter count** to fit within embedded memory budgets
- **Quantization-aware** design for post-training quantization

---

## Pipeline

```
Raw ECG Signal
      ↓
preprocessing.py  →  Filtering, normalization, segmentation
      ↓
model.py          →  CNN classifier
      ↓
main.py           →  Training loop, evaluation, model export
      ↓
ecg_model.onnx    →  ONNX export
      ↓
Arduino Nano 33 BLE  →  Edge inference (via TFLite / C array)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Arduino Nano 33 BLE (for embedded deployment)
- Arduino IDE with TensorFlow Lite Micro library

### Installation

```bash
git clone https://github.com/adii11001/Arrhythmia-detection-model.git
cd Arrhythmia-detection-model
pip install -r requirements.txt
```

### Training

```bash
python main.py
```

The best model weights will be saved to `best_ecg_model.pth` and the ONNX export to `ecg_model.onnx`.

---

## Deployment on Arduino Nano 33 BLE

To deploy on the Arduino Nano 33 BLE:

1. Convert `ecg_model.onnx` to TensorFlow Lite format using ONNX → TF → TFLite toolchain.
2. Apply **post-training quantization** (INT8) to reduce model size.
3. Convert the `.tflite` model to a C byte array using `xxd`:
   ```bash
   xxd -i ecg_model.tflite > ecg_model_data.cc
   ```
4. Include the generated C array in your Arduino sketch alongside the TensorFlow Lite Micro library.
5. Flash to the Arduino Nano 33 BLE via Arduino IDE.

---

## Requirements

Key Python dependencies (see `requirements.txt` for full list):

- `torch` — Model training
- `onnx` / `onnxruntime` — Model export and validation
- `numpy` — Signal processing
- `scipy` — ECG filtering utilities
- `scikit-learn` — Evaluation metrics

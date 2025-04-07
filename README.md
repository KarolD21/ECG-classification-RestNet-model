# 🩺 PTB-XL ECG Signal Processing & Classification

This repository processes ECG signal data from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), a large clinical 12-lead ECG database published on PhysioNet. The pipeline includes data loading, preprocessing, and feeding signals into a deep learning model for multi-label classification of cardiac conditions.

---

## 📂 Files Description

- **`data_input.py`**  
  Handles loading of raw ECG signal data and associated labels from the PTB-XL database.  
  **Key functionalities:**
  - Extracts signal arrays and structured diagnostic labels.
  - Formats input into NumPy arrays or tensors ready for processing.

- **`data_processing.py`**  
  Contains utilities to preprocess ECG signals for model training or inference.  
  **Key functionalities:**
  - Normalization and resampling of ECG signals.
  - Optional signal augmentation (e.g., noise, scaling).
  - Label binarization and encoding for multi-label classification tasks.

- **`model.py`**  
  Defines the neural network architecture used to analyze ECG signals.  
  **Key functionalities:**
  - Implements a deep learning model (e.g., CNN or RNN) for multi-label classification.
  - Handles model initialization, forward pass, and optionally custom loss functions.

---

## 🧪 Input Format

- **Input Signals:** Raw 12-lead ECG signals, each sampled at 500Hz, typically 10 seconds in duration.
- **Labels:** Diagnostic statements (e.g., myocardial infarction, atrial fibrillation) mapped to multi-hot encoded vectors for training.

---

## 🧠 Model Architecture

The deep learning model implemented in `model.py` is designed for **multi-label classification of ECG signals** from the PTB-XL dataset. It takes preprocessed 1D ECG waveforms as input and outputs a probability vector representing the presence of various cardiac conditions.

### 🔧 Key Features

- **Input Shape**:  
  12-lead ECG signals, typically shaped as `(batch_size, 12, 5000)`  
  (10 seconds per sample at 500 Hz sampling rate)

- **Architecture**:  
  A custom neural network built using:
  - **1D Convolutional Layers** to extract temporal and spatial features from the waveforms
  - **Batch Normalization** and **ReLU** activation to stabilize and accelerate training
  - **Global Average Pooling** or **Flattening** to reduce dimensionality
  - **Fully Connected Layers** for final classification over multiple diagnostic labels

- **Output**:  
  A vector of logits representing the likelihood of each diagnostic class.  
  Apply `sigmoid` to convert the logits to probability scores for each label.

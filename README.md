# Histopathologic Cancer Detection: Transfer Learning & Optimization

## 🌟 Overview
This project addresses the Kaggle competition "Histopathologic Cancer Detection," a binary image classification task. The goal is to classify small, $96 \times 96$ pixel patches of lymph node tissue to determine the presence (label 1) or absence (label 0) of invasive ductal carcinoma (breast cancer).

The notebook documents the journey from a simple baseline to a transfer learning approach, focusing heavily on **performance optimization** to overcome the challenges of a large dataset and resource constraints in a notebook environment.

## 🎯 Goal
To build a Convolutional Neural Network (CNN) capable of distinguishing cancer tissue in image patches and evaluate performance using the **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**.

## 📁 Data Summary
* **Source:** Kaggle Competition: Histopathologic Cancer Detection.
* **Data Size:** Over 220,000 image patches (`.tif` files).
* **Input:** $96 \times 96 \times 3$ (RGB) image patches.
* **Target:** Binary classification (0: No Cancer, 1: Cancer).

## 💡 Methodology

### 1. Baseline Model (Simple CNN)
A custom, shallow CNN was built and trained on the full dataset as a baseline.
* **Architecture:** 3 Convolutional Blocks (Conv2D $\rightarrow$ Max Pooling) followed by a Dropout/Dense classification head.
* **Baseline Result:** Achieved a **Validation AUC of 0.9365** on the full dataset.

### 2. Optimized Transfer Learning Model
To improve accuracy and speed up convergence, the primary approach shifted to **Transfer Learning**.

* **Base Model:** **ResNet50**, pre-trained on ImageNet weights.
* **Strategy:** Weights were **frozen** to prevent catastrophic forgetting, and a new custom classification head was trained on the pathology data.

### 3. Hyperparameter Tuning and Sampling
A limited **Grid Search** was performed on the sampled data, testing combinations of Learning Rate and Batch Size.

| Model | Batch Size | Learning Rate | Best Validation AUC |
| :--- | :--- | :--- | :--- |
| Simple CNN (Sampled) | 64 | N/A | **0.9300** |
| ResNet50 TL (Sampled) | 64 | 0.00010 | **0.7237** |
| *Note: The drop in TL AUC suggests the aggressive sampling hurt feature generalization for the frozen ResNet layers.* |

## 🚀 Performance Optimization Triumphs

Due to severe resource constraints and slow training times, significant effort was dedicated to optimizing the execution environment. This section summarizes the techniques implemented to achieve functional training times:

| Optimization Implemented | Problem Solved | Impact |
| :--- | :--- | :--- |
| **Data Sampling (5% of data)** | Too many steps per epoch ($\approx 2900$ steps) | **10x Speedup** in epoch time. |
| **GPU Memory Growth & cuDNN Fix** | TensorFlow failing to use optimized GPU algorithms (`slow_operation_alarm` warnings). | **Massive Speedup** (fixed $\approx 679 \text{ms/step}$ bottleneck). |
| **Multiprocessing Workers** | CPU bottleneck in data loading (`workers=8`, `use_multiprocessing=True`). | **Reduced GPU idle time.** |
| **Set `verbose=2` in `model.fit()`** | Browser crashing due to "Out of Memory" from rendering excessive log output. | **Prevented browser crash.** |
| **Mixed Precision** | General GPU computation time. | **~50% reduction** in computation time. |

## 🏆 Final Results and Conclusion

The Simple CNN model, when trained thoroughly on the full data, was the most performant model, demonstrating a strong ability to classify the image patches.

* **Best Validation AUC (Simple CNN, Full Data):** **0.9365**
* **Key Learning:** While Transfer Learning is the standard, achieving reliable performance requires sufficient data. The aggressive data sampling needed for quick iteration compromised the performance of the pre-trained ResNet model, leading to lower scores compared to the custom CNN.

Future work would involve dedicated fine-tuning of the ResNet model with a larger dataset and further exploring techniques like Test-Time Augmentation (TTA) and model ensembling.

# 🍅 Tomato Leaf Disease Classification with CNN

Deep Learning project to **automatically classify tomato leaf diseases from images** using a **Convolutional Neural Network (CNN) built from scratch in PyTorch**.

The goal is to detect plant diseases early using computer vision, helping farmers take faster and more informed agronomic decisions.

🔗 Repository:
[https://github.com/isabellaperezcav/tomato-leaf-disease-classification-cnn](https://github.com/isabellaperezcav/tomato-leaf-disease-classification-cnn)

---

# 🌱 Project Context

Leaf diseases are responsible for **millions of dollars in crop losses worldwide**. Early detection allows:

* Faster intervention
* Reduced fungicide use
* Better crop management decisions

This project develops a **Deep Learning model capable of identifying the health condition of a tomato leaf from an image.**

The model classifies leaves into **four possible categories**.

---

# 🧠 Classes to Classify

| Class                            | Type                    | Description                               |
| -------------------------------- | ----------------------- | ----------------------------------------- |
| 🌿 Tomato_Healthy                | Healthy                 | Leaf without visible symptoms             |
| 🟤 Tomato_Early_Blight           | Fungal – *Alternaria*   | Dark concentric spots with yellow borders |
| ⚫ Tomato_Late_Blight             | Fungal – *Phytophthora* | Brown-black water-soaked lesions          |
| 🟡 Tomato_Yellow_Leaf_Curl_Virus | Virus – TYLCV           | Yellowing, curling and deformation        |

---

# ⚠️ Class Imbalance

The dataset contains **~2× more images of TYLCV** than the other classes.

Because of this, **Accuracy is not a reliable metric**.

Example problem:

* A model predicting only TYLCV could reach ~40% accuracy
* But would fail on all other classes.

For this reason, the **main evaluation metric is Macro F1-Score**, which treats all classes equally.

---

# 📊 Evaluation Metric

Macro F1 Score is computed as:

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, average='macro')
```

Performance levels:

| Level         | F1 Macro    | Performance             |
| ------------- | ----------- | ----------------------- |
| ⚠️ Minimum    | 0.80 – 0.85 | Basic functioning model |
| 🎯 Target     | 0.86 – 0.92 | Good generalization     |
| 🏆 Excellence | > 0.93      | High-performing model   |

---

# 🧪 Technical Restrictions

## 🚫 Not Allowed

The model **must be built from scratch**, therefore the following were **not allowed**:

* Transfer Learning (ResNet, EfficientNet, VGG pretrained, MobileNet, CLIP)
* Pretrained models from `timm` or `keras.applications`
* AutoML frameworks
* Architectures copied directly from research papers

---

## ✅ Allowed

The model was implemented using **pure PyTorch layers**, including:

* `nn.Conv2d`
* `nn.BatchNorm2d`
* `nn.ReLU`
* `nn.MaxPool2d`
* `Dropout`
* `Data Augmentation`
* Class balancing strategies

Regularization techniques used include:

* Batch Normalization
* Dropout
* Data Augmentation
* Weighted sampling for class imbalance

---

# 🏗 Model Architecture

The model follows a **custom CNN architecture inspired by VGG-style convolutional blocks**, including:

* Convolutional blocks with BatchNorm
* Stacked convolution layers
* MaxPooling for spatial reduction
* Dropout for regularization
* Fully connected classifier

Typical pipeline:

```
Input Image (RGB 128x128)
        │
Conv Block
        │
Conv Block
        │
Conv Block
        │
Conv Block
        │
Adaptive Pooling
        │
Fully Connected Layers
        │
Softmax Classification
```

---

# 🖼 Input Preprocessing

Images were:

* Resized to **128×128**
* Converted to **RGB**
* Normalized using ImageNet statistics:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# 🔄 Data Augmentation

Applied only during training:

* Random horizontal flip
* Random rotation
* Color jitter
* Random crop

This improves **generalization and robustness**.

---

# 📦 Dataset

Student dataset split:

| Class                         | Total Images | Train     | Validation |
| ----------------------------- | ------------ | --------- | ---------- |
| Tomato_Healthy                | 1432         | ~1146     | ~286       |
| Tomato_Early_Blight           | 900          | ~720      | ~180       |
| Tomato_Late_Blight            | 1719         | ~1375     | ~344       |
| Tomato_Yellow_Leaf_Curl_Virus | 2888         | ~2310     | ~578       |
| **TOTAL**                     | **6939**     | **~5551** | **~1388**  |

---

# 📈 Model Evaluation

The project includes several evaluation visualizations:

* Class distribution
* Dataset splits
* Confusion matrix
* Training curves
* F1 score per class
* Error analysis

Example outputs included in the repository:

* `curvas_entrenamiento.png`
* `matriz_confusion.png`
* `f1_por_clase.png`
* `errores_analisis.png`

---

# 📂 Repository Structure

```
tomato-leaf-disease-classification-cnn
│
├── Sprint1_DL.ipynb
├── evaluador_sprint1.py
├── .gitignore
│
├── curvas_entrenamiento.png
├── matriz_confusion.png
├── f1_por_clase.png
├── distribucion_clases.png
├── distribucion_split.png
├── exploracion_muestras.png
└── errores_analisis.png
```

---

# 🚀 How to Run

1️⃣ Clone the repository

```bash
git clone https://github.com/isabellaperezcav/tomato-leaf-disease-classification-cnn.git
```

2️⃣ Install dependencies

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
```

3️⃣ Open the notebook

```
Sprint1_DL.ipynb
```

4️⃣ Download dataset (provided in course instructions)

---

# 🎯 Project Goals

* Build a CNN **without transfer learning**
* Handle **class imbalance**
* Optimize **F1 Macro score**
* Analyze model errors and performance

---

# 📚 Technologies

* Python
* PyTorch
* Scikit-learn
* Matplotlib
* Google Colab

---

# 👩‍💻 Author

**Isabella Pérez**

Deep Learning / Machine Learning project focused on **computer vision for agricultural disease detection**.

ecto de clase → proyecto de portafolio profesional**.

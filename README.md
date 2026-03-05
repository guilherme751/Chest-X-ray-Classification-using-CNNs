# 🏥 Chest X-ray Classification using CNNs

A comprehensive deep learning project for medical image classification using Convolutional Neural Networks (CNNs) to detect COVID-19, Pneumonia, and Normal conditions from chest X-ray images.

> **Note:** This was a course assignment for a university discipline. The original assignment guidelines are in Portuguese, which is why some comments and documentation in the notebook may be in Portuguese.

## 📋 Project Overview

This project trains three different CNN architectures to classify chest X-ray images into three categories:
- **COVID-19** - Patients with COVID-19 infection
- **Pneumonia** - Patients with pneumonia
- **Normal** - Healthy patients

The project implements and compares three different training approaches:
1. **Custom CNN** - A CNN architecture designed from scratch
2. **Pre-trained with Frozen Layers** - Transfer learning with frozen convolutional layers
3. **Fine-tuned Pre-trained** - Full model fine-tuning on the medical dataset

## 🎯 Objectives

- Implement a custom CNN architecture for medical image classification
- Apply transfer learning using pre-trained models (e.g., ResNet, VGG, EfficientNet)
- Compare frozen vs. fine-tuning strategies for transfer learning
- Implement data augmentation techniques for medical imaging
- Evaluate model performance on test data
- (Bonus) Use explainability methods (GradCAM++) to visualize decision-making

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **PyTorch** - Deep learning framework (no high-level wrappers like Keras)
- **Torchvision** - Image transformations and pre-trained models
- **Kagglehub** - Dataset downloading
- **PIL/Pillow** - Image loading and preprocessing
- **Pandas** - Data manipulation
- **Scikit-learn** - Train/validation split
- **Matplotlib/Seaborn** - Visualization
- **GradCAM++** (Bonus) - Model explainability

## 📊 Dataset

**Source:** [Chest X-ray Image (COVID19, PNEUMONIA, and NORMAL) on Kaggle](https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image)

**Characteristics:**
- Multi-format images: RGB and grayscale
- Variable bit depths: 8-bit and 16-bit
- Different file formats: JPEG, BITMAP
- Variable resolutions
- Pre-split into train and test sets

**Data Split:**
- Training set: ~80% (further split into train/validation)
- Validation set: ~20% of training data
- Test set: Provided separately

**Classes:**
- COVID-19
- Pneumonia  
- Normal (Healthy)

## 🏗️ Architecture

### 1. Custom CNN
A CNN architecture designed from scratch with:
- Multiple convolutional layers
- Batch normalization
- Max pooling
- Dropout for regularization
- Fully connected classifier

### 2. Pre-trained Model (Frozen)
Transfer learning approach:
- Load pre-trained model (ResNet/VGG/EfficientNet)
- Freeze convolutional layers
- Replace classifier head
- Train only the new classifier

### 3. Pre-trained Model (Fine-tuned)
Full fine-tuning approach:
- Load pre-trained model
- Unfreeze all layers
- Replace classifier head
- Train entire network end-to-end

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision kagglehub pillow pandas scikit-learn matplotlib seaborn
```

For explainability (bonus):
```bash
pip install grad-cam
```

### Running the Project

1. **Open the notebook:**
   ```bash
   jupyter notebook T2_Redes_Neurais_CNN.ipynb
   ```

2. **Run cells sequentially** - The notebook will:
   - Download the chest X-ray dataset automatically
   - Explore and visualize the data
   - Create train/validation/test datasets
   - Apply data augmentation
   - Train three CNN models
   - Compare performance metrics
   - Generate visualization plots

### GPU Support

The code is designed to use GPU acceleration:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

Works on:
- Google Colab (free GPU)
- Local GPU (NVIDIA with CUDA)
- CPU (slower training)

## 📈 Data Preprocessing Pipeline

### Image Loading
```python
def load_img(path):
    # Handles diverse formats (RGB, grayscale, 8-bit, 16-bit)
    img = Image.open(path).convert('RGB')  # Ensure 3 channels
    img = v2.functional.to_image(img)
    img = v2.functional.to_dtype(img, dtype=torch.uint8, scale=True)
    return img
```

### Data Augmentation (Training Only)
```python
train_transform = v2.Compose([    
    v2.RandomResizedCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.RandomPerspective(0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

### Validation/Test Transform
```python
val_test_transform = v2.Compose([
    v2.Resize(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
```

## 🔬 Training Process

### Custom Dataset Class
```python
class ChestXray(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx, :]
        img = load_img(img_path)
        tensor = self.transforms(img)
        return tensor, label
```

### Training Loop Structure
1. Forward pass through the model
2. Compute loss (CrossEntropyLoss)
3. Backward propagation
4. Optimizer step (Adam/SGD)
5. Track training metrics
6. Validate on validation set
7. Save best model checkpoint

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Detailed error analysis
- **Training/Validation Curves**: Loss and accuracy over epochs

## 🎓 Assignment Details

**Course Assignment Specifications:**
- Deadline: December 23, 2025
- Submission: Via Testr system
- Points: 10 points (+ 1 bonus point)
- Individual work

**Requirements Met:**
- ✅ Three different CNN models trained
- ✅ Custom architecture implemented from scratch
- ✅ Pre-trained model with frozen layers
- ✅ Pre-trained model with fine-tuning
- ✅ Proper train/validation/test split
- ✅ GPU-enabled code (compatible with Colab)
- ✅ Pure PyTorch implementation (no Keras/Lightning)
- ✅ Data augmentation on training set only
- ✅ Comparative performance analysis with plots

**Bonus Implementation:**
- ✅ GradCAM++ for model explainability and decision visualization

## 📊 Model Comparison

The notebook includes comparative analysis across all three models:

| Model | Architecture | Trainable Params | Train Acc | Val Acc | Test Acc |
|-------|--------------|------------------|-----------|---------|----------|
| Custom CNN | From scratch | ~X million | TBD | TBD | TBD |
| Pre-trained (Frozen) | ResNet/VGG + Classifier | ~X thousand | TBD | TBD | TBD |
| Pre-trained (Fine-tuned) | Full ResNet/VGG | ~X million | TBD | TBD | TBD |

## 🔍 Explainability (Bonus)

Using GradCAM++ to visualize which regions of the X-ray influenced the model's decision:

```python
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

# Initialize GradCAM++
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

# Generate heatmap
grayscale_cam = cam(input_tensor=image)

# Overlay on original image
visualization = show_cam_on_image(rgb_img, grayscale_cam)
```

This helps:
- Verify the model is looking at relevant regions (lungs, not artifacts)
- Build trust in medical AI decisions
- Debug model behavior
- Provide interpretability for clinical use

## 🎯 Key Features

- **Medical Image Handling**: Robust preprocessing for diverse X-ray formats
- **Data Augmentation**: Realistic augmentations for medical imaging
- **Transfer Learning**: Leveraging pre-trained ImageNet models
- **Model Comparison**: Systematic evaluation of different approaches
- **GPU Optimization**: Efficient training with CUDA support
- **Reproducibility**: Fixed random seeds for consistent results
- **Visualization**: Comprehensive plots for analysis

## 📁 Project Structure

```
.
├── T2_Redes_Neurais_CNN.ipynb    # Main notebook
└── README.md                      # This file
```

**Generated during execution:**
- Downloaded dataset from Kaggle
- Trained model checkpoints (.pth files)
- Training history plots
- Confusion matrices
- GradCAM++ visualizations (bonus)

## 🔧 Implementation Tips

### Handling Image Diversity
The dataset contains images with varying properties. Key preprocessing steps:
1. Convert all to RGB (3 channels)
2. Normalize bit depth to 8-bit
3. Resize to consistent dimensions (224×224)
4. Apply normalization for neural network input

### Data Augmentation Strategy
For medical images, use augmentations that preserve diagnostic features:
- ✅ Horizontal flips (chest anatomy is symmetric)
- ✅ Small rotations (±30°)
- ✅ Slight perspective changes
- ❌ Avoid extreme distortions
- ❌ No color jittering (X-rays are grayscale-origin)

### Transfer Learning Best Practices
**Frozen layers:**
- Faster training
- Requires less data
- Lower risk of overfitting
- May miss domain-specific features

**Fine-tuning:**
- Longer training time
- Better adaptation to medical domain
- Higher capacity model
- Requires careful learning rate selection

## 📚 Medical Imaging Context

### Why This Matters
- **COVID-19 Detection**: Rapid screening tool during pandemic
- **Pneumonia Diagnosis**: Common and serious respiratory condition
- **AI in Healthcare**: Growing field with real-world impact
- **Diagnostic Support**: Assists radiologists in decision-making

### Clinical Considerations
- Models should be validated by medical professionals
- High precision/recall needed to avoid misdiagnosis
- Explainability crucial for clinical adoption
- Must handle diverse X-ray acquisition protocols

## 🏆 Results Summary

The project demonstrates:
- Effectiveness of CNNs for medical image classification
- Trade-offs between custom and pre-trained architectures
- Impact of fine-tuning vs. frozen features
- Importance of data augmentation in limited medical datasets

Detailed results with accuracy metrics, confusion matrices, and training curves are available in the notebook.

## 📚 References

**Dataset:**
- [Chest X-ray Image Dataset on Kaggle](https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image)

**Technical Resources:**
- [PyTorch ImageFolder Tutorial](https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/)
- [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [Transfer Learning with PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

**Medical Imaging:**
- Deep Learning for Medical Image Analysis
- CNN Architectures for Healthcare Applications
- Explainable AI in Medical Diagnosis

## 💡 Learning Outcomes

After completing this project, you will understand:
- How to build CNNs from scratch for classification tasks
- Transfer learning strategies and when to use them
- Medical image preprocessing challenges and solutions
- Data augmentation techniques for healthcare data
- Model evaluation and comparison methodologies
- Explainability methods for deep learning models

## 🔒 Ethical Considerations

This is an educational project. For real clinical applications:
- Extensive validation required
- Regulatory approval necessary (FDA, CE marking)
- Medical professional oversight essential
- Patient privacy and data security critical
- Bias and fairness must be addressed

## 🤝 Contributing

This was an academic project completed as part of coursework. Feel free to fork and extend with additional architectures or datasets.

## 📄 License

This project is part of academic coursework. The dataset is available under Kaggle's terms of use.

---

**Developed as part of university coursework - 2025**

**Note:** This project is for educational purposes only and should not be used for actual medical diagnosis without proper validation and regulatory approval.

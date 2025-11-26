## Traffic Sign Classification using Deep Learning

A convolutional neural network (CNN) implementation for classifying German traffic signs from the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset.

## Overview

This project implements a CNN-based traffic sign classifier trained on 34,799 32x32 RGB images representing 43 different traffic sign classes. The model achieves **94.9% validation accuracy** and **99.2% training accuracy** on the German Traffic Sign Dataset.

## Project Results

- **Validation Accuracy**: 94.9%
- **Test Accuracy**: ~99.2% (on training set after convergence)
- **Model Type**: 5-layer Convolutional Neural Network with dropout regularization
- **Optimizer**: Adam optimizer with learning rate 0.001
- **Training Epochs**: 60 (with early stopping after 10 epochs without improvement)

## Project Structure

```
├── Traffic_Sign_Classifier.ipynb  # Main Jupyter notebook with implementation
├── Traffic_Sign_Classifier.html   # Exported notebook as HTML
├── Traffic_Sign_Classifier_Report.pdf  # Detailed project report
├── signnames.csv                  # Mapping of class IDs to traffic sign names
├── new_sign/                      # Test images from web (traffic1.jpg - traffic5.jpg)
├── lenet.meta                     # Saved model metadata
├── lenet.index                    # Saved model index
├── lenet.data-*                   # Saved model weights
└── README.md                      # This file
```

## Model Architecture

The model uses a modified LeNet-5 architecture with the following layers:

1. **Convolutional Layer 1**: Input 32×32×1 → Output 28×28×16 (5×5 filters)
   - ReLU activation
   - Max pooling (2×2) → Output 14×14×16

2. **Convolutional Layer 2**: Input 14×14×16 → Output 10×10×32 (5×5 filters)
   - ReLU activation
   - Max pooling (2×2) → Output 5×5×32

3. **Flatten**: Input 5×5×32 → Output 800

4. **Fully Connected Layer 1**: Input 800 → Output 120
   - ReLU activation
   - Dropout (keep_prob: 0.5)

5. **Fully Connected Layer 2**: Input 120 → Output 84
   - ReLU activation
   - Dropout (keep_prob: 0.5)

6. **Output Layer**: Input 84 → Output 43 (traffic sign classes)

## Data Preprocessing

- **Grayscale Conversion**: RGB images converted to grayscale (color information is not critical for traffic sign classification)
- **Normalization**: Pixel values normalized to range [-0.5, 0.5]
- **Data Balancing**: Training data balanced by duplicating underrepresented classes to achieve uniform distribution (2,010 samples per class)

## Training Strategy

1. **Dataset**: 34,799 training samples, 4,410 validation samples, 12,630 test samples
2. **Batch Size**: 128
3. **Learning Rate**: 0.001
4. **Dropout**: 0.5 keep probability to prevent overfitting
5. **Early Stopping**: Stops after 10 epochs without validation accuracy improvement

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 1.x
- NumPy
- OpenCV (cv2)
- Pandas
- Matplotlib
- scikit-learn

### Running the Notebook

```bash
jupyter notebook Traffic_Sign_Classifier.ipynb
```

The notebook will:
1. Load and explore the German Traffic Sign Dataset
2. Preprocess images (grayscale conversion, normalization)
3. Balance the training dataset
4. Build and train the CNN model
5. Evaluate on test images
6. Make predictions on new traffic sign images

## Key Implementation Details

### Handling Class Imbalance

The original dataset had uneven class distributions. Data was balanced by duplicating training examples from underrepresented classes until all classes had 2,010 samples, bringing total training samples to 86,430.

### Overcoming Overfitting

Initial attempts with standard LeNet-5 showed high training accuracy (99%) but low validation accuracy, indicating overfitting. Solutions implemented:

- Added dropout layers (0.5 keep probability) in fully connected layers
- Increased convolutional filters (16 and 32 vs. original 6 and 16)
- Used grayscale preprocessing to reduce input dimensionality
- Applied data augmentation through balancing

### New Traffic Sign Testing

The model was tested on 5 traffic signs from German street views:
- **Traffic1.jpg** (Speed Limit 20): 99% confidence
- **Traffic2.jpg** (Speed Limit 70): 95% confidence
- **Traffic3.jpg** (Road Work): 99% confidence
- **Traffic4.jpg** (Speed Limit 120): 99% confidence
- **Traffic5.jpg** (Keep Right): 98% confidence

## Performance Metrics by Training Progress

| Epoch | Training Acc | Validation Acc |
|-------|-------------|----------------|
| 1     | 73.9%       | 64.8%          |
| 5     | 93.7%       | 87.0%          |
| 10    | 96.7%       | 91.1%          |
| 20    | 98.1%       | 92.1%          |
| 30    | 98.6%       | 94.1%          |
| 40    | 98.9%       | 94.6%          |
| 50    | 99.1%       | 94.1%          |
| 60    | 99.2%       | 94.9%          |

## Results

The model successfully classifies German traffic signs with high accuracy. The implementation demonstrates:

- Effective use of convolutional neural networks for image classification
- Data preprocessing and balancing techniques
- Regularization strategies (dropout) to prevent overfitting
- Transfer learning principles adapted for traffic sign recognition

## References

- [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- [Sermanet et al. - Traffic Sign Recognition Paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
- [LeNet-5 Architecture](http://yann.lecun.com/exdb/lenet/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

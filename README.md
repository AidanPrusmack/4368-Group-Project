# 4368-Group-Project
# Chest X-Ray Pneumonia Classification

This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Normal** or **Abnormal** (indicative of Pneumonia). It includes preprocessing, augmentation, model training, evaluation, and visualization of performance metrics.

## Dataset

The dataset is the [Coronahack Chest X-Ray Dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset), which consists of labeled chest X-ray images divided into training and test sets.

- `Chest_xray_Corona_Metadata.csv`: Metadata containing labels and dataset splits.
- Images are located in:
  - `Coronahack-Chest-XRay-Dataset/train/`
  - `Coronahack-Chest-XRay-Dataset/test/`

## Features

- Custom CNN model with BatchNormalization and Dropout for regularization
- Image preprocessing and augmentation using `ImageDataGenerator`
- Class weight balancing to address class imbalance
- Training callbacks: EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
- Evaluation metrics: Accuracy, ROC AUC, Confusion Matrix, Precision, Recall, F1
- Visualization of:
  - Training curves (`training_curves.png`)
  - Confusion matrix (`confusion_matrix.png`)
  - ROC curve (`roc_curve.png`)
- Saves trained model as `chest_xray_model.keras`

## Setup

1. Clone the repository and place the dataset in the `dataset/` directory.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = load_model('chest_xray_model.keras')

test_csv_path = "dataset/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv"
test_image_dir = "dataset/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"

def load_test_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    images = []
    true_labels = []
    image_paths = []

    for _, row in df.iterrows():
        path = os.path.join(image_dir, row['X_ray_image_name'])
        if not os.path.exists(path):
            continue
        image_paths.append(path)
        true_labels.append('Abnormal' if row['Label'] == 'Pnemonia' else 'Normal')


    return image_paths, true_labels


test_image_paths, test_true_labels = load_test_data(test_csv_path, test_image_dir)

results = []
for img_path, true_label in zip(test_image_paths, test_true_labels):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_label = "Normal" if np.argmax(prediction[0]) == 0 else "Abnormal"
    confidence = float(prediction[0][np.argmax(prediction[0])])

    is_correct = predicted_label == true_label
    results.append({
        'image': os.path.basename(img_path),
        'true_label': true_label,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'correct': is_correct
    })

    if not is_correct:
        print(
            f"Misclassified: {os.path.basename(img_path)} - True: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.2f}")

results_df = pd.DataFrame(results)
accuracy = results_df['correct'].mean()

print(f"Overall accuracy: {accuracy:.4f}")
print(f"Total correct: {results_df['correct'].sum()} out of {len(results_df)}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img


def plot_confusion_matrix(results_df):
    cm = confusion_matrix(
        (results_df['true_label'] == 'Abnormal').astype(int),
        (results_df['predicted_label'] == 'Abnormal').astype(int)
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_roc_curve(results_df):
    y_true = (results_df['true_label'] == 'Abnormal').astype(int)

    y_scores = np.array([
        conf if pred == 'Abnormal' else 1 - conf
        for conf, pred in zip(results_df['confidence'], results_df['predicted_label'])
    ])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

    return roc_auc


def plot_precision_recall_curve(results_df):
    y_true = (results_df['true_label'] == 'Abnormal').astype(int)

    y_scores = np.array([
        conf if pred == 'Abnormal' else 1 - conf
        for conf, pred in zip(results_df['confidence'], results_df['predicted_label'])
    ])

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.show()

def plot_confidence_distribution(results_df):
    plt.figure(figsize=(10, 6))

    correct_conf = results_df[results_df['correct']]['confidence']
    incorrect_conf = results_df[~results_df['correct']]['confidence']

    plt.hist(correct_conf, alpha=0.5, bins=20, label='Correct Predictions', color='green')
    plt.hist(incorrect_conf, alpha=0.5, bins=20, label='Incorrect Predictions', color='red')

    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_distribution.png')
    plt.show()


def plot_class_distribution(results_df):
    class_counts = results_df['true_label'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Test Set')
    for i, count in enumerate(class_counts.values):
        plt.text(i, count + 5, str(count), ha='center')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

def plot_accuracy_by_class(results_df):
    class_accuracy = results_df.groupby('true_label')['correct'].mean()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_accuracy.index, y=class_accuracy.values)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Class')
    plt.ylim(0, 1)
    for i, acc in enumerate(class_accuracy.values):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('accuracy_by_class.png')
    plt.show()

def display_misclassified_examples(results_df, test_image_paths, num_samples=3):
    misclassified = results_df[~results_df['correct']]

    if len(misclassified) == 0:
        print("No misclassified images found.")
        return

    samples = min(num_samples, len(misclassified))
    sampled_indices = random.sample(range(len(misclassified)), samples)

    fig, axes = plt.subplots(1, samples, figsize=(5 * samples, 5))
    if samples == 1:
        axes = [axes]

    for i, idx in enumerate(sampled_indices):
        misclassified_row = misclassified.iloc[idx]
        img_name = misclassified_row['image']

        img_path = None
        for path in test_image_paths:
            if os.path.basename(path) == img_name:
                img_path = path
                break

        if img_path:
            img = load_img(img_path, target_size=(224, 224))
            axes[i].imshow(img)
            axes[i].set_title(f"True: {misclassified_row['true_label']}\n"
                              f"Pred: {misclassified_row['predicted_label']}\n"
                              f"Conf: {misclassified_row['confidence']:.2f}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.show()

def plot_threshold_performance(results_df):
    thresholds = np.arange(0.5, 1.0, 0.05)
    accuracies = []

    for threshold in thresholds:
        high_conf_preds = results_df[results_df['confidence'] >= threshold]
        if len(high_conf_preds) > 0:
            accuracy = high_conf_preds['correct'].mean()
        else:
            accuracy = np.nan
        accuracies.append(accuracy)

    coverage = [(results_df['confidence'] >= t).mean() for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(thresholds, accuracies, 'o-', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Coverage (% of dataset)', color='tab:red')
    ax2.plot(thresholds, np.array(coverage) * 100, 'o-', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Accuracy vs. Coverage at Different Confidence Thresholds')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('threshold_performance.png')
    plt.show()


def generate_all_visualizations(results_df, test_image_paths):
    print("Generating visualizations...")
    plot_confusion_matrix(results_df)
    roc_auc = plot_roc_curve(results_df)
    plot_precision_recall_curve(results_df)
    plot_confidence_distribution(results_df)
    plot_class_distribution(results_df)
    plot_accuracy_by_class(results_df)
    display_misclassified_examples(results_df, test_image_paths)
    plot_threshold_performance(results_df)

    y_true = (results_df['true_label'] == 'Abnormal').astype(int)
    y_pred = (results_df['predicted_label'] == 'Abnormal').astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal']))
    print(f"ROC AUC: {roc_auc:.4f}")

generate_all_visualizations(results_df, test_image_paths)

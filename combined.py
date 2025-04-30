import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, Input, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths to dataset
csv_path = "dataset/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv"
base_img_dir = "dataset/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
train_img_dir = os.path.join(base_img_dir, "train/")
test_img_dir = os.path.join(base_img_dir, "test/")


def create_model():
    """Create and compile an improved CNN model with regularization"""
    model = models.Sequential([
        Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Removed the precision and recall metrics that were causing issues
    )
    return model


def load_and_preprocess_data(df, image_dir, is_training=False):
    """Load and preprocess images from DataFrame"""
    images, labels, image_paths = [], [], []

    for _, row in df.iterrows():
        path = os.path.join(image_dir, row['X_ray_image_name'])
        if not os.path.exists(path):
            continue

        if row['Label'] == 'Normal':
            label = 0
        else:
            label = 1

        try:
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
            image_paths.append(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    return np.array(images), np.array(labels), image_paths


def visualize_results(y_true, y_pred, class_names):
    """Generate and print confusion matrix and classification report"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Print in a readable format
    print(f"\n{class_names[0]} correctly identified: {cm[0, 0]}")
    print(f"{class_names[0]} incorrectly classified as {class_names[1]}: {cm[0, 1]}")
    print(f"{class_names[1]} correctly identified: {cm[1, 1]}")
    print(f"{class_names[1]} incorrectly classified as {class_names[0]}: {cm[1, 0]}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)


# Main execution
if __name__ == "__main__":
    # Load the full dataset
    df = pd.read_csv(csv_path)

    # Split based on the Dataset_type column, not randomly
    train_df = df[df['Dataset_type'] == 'TRAIN']
    test_df = df[df['Dataset_type'] == 'TEST']

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    # Class distribution
    print("\nTraining class distribution:")
    print(train_df['Label'].value_counts())
    print("\nTesting class distribution:")
    print(test_df['Label'].value_counts())

    # Load and preprocess data
    x_train, y_train, _ = load_and_preprocess_data(train_df, train_img_dir)
    x_test, y_test, test_paths = load_and_preprocess_data(test_df, test_img_dir)

    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Create and train the model
    model = create_model()
    print(model.summary())

    # Add callbacks for better training
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        ),
        callbacks.ModelCheckpoint(
            filepath='best_chest_xray_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    class_weights = {0: len(train_df) / len(train_df[train_df['Label'] == 'Normal']),
                     1: len(train_df) / len(train_df[train_df['Label'] == 'Pnemonia'])}
    # Fixed: Don't use the generator directly through steps_per_epoch
    # Instead, fit directly with the augmented data
    batch_size = 32
    model.fit(
        x_train, y_train,  # Use x_train and y_train directly
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        class_weight=class_weights
    )

    # Evaluate on test set
    print("\nEvaluating model on test set:")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    # Get predictions for more detailed metrics
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate AUC if possible (requires probabilities)
    try:
        auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        print(f"ROC AUC: {auc:.4f}")
    except:
        print("Could not calculate ROC AUC")

    # Visualize results
    class_names = ['Normal', 'Abnormal']
    visualize_results(y_test, y_pred, class_names)

    # Save the trained model
    model.save('chest_xray_model.keras')
    print("Model saved as 'chest_xray_model.keras'")

    # Check for misclassified images
    print("\nMisclassified Images:")
    misclassified_count = 0
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            true_label = class_names[y_test[i]]
            pred_label = class_names[y_pred[i]]
            confidence = y_pred_prob[i][y_pred[i]]
            image_name = os.path.basename(test_paths[i])
            print(f"Image: {image_name}, True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
            misclassified_count += 1

            # Limit display of misclassified to keep output manageable
            if misclassified_count >= 20:
                print("(More misclassified images exist but not shown for brevity)")
                break

    # Calculate precision and recall manually (since we removed the metrics)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nManually calculated metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

tf.keras.utils.plot_model(model, to_file="model_architecture.png", show_shapes=True)
import matplotlib.pyplot as plt

history = model.history.history

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Normal', 'Abnormal'], cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

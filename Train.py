import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from PIL import Image


def create_model():
    model = models.Sequential([
        Input(shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def predict_xray(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)
    result = "Normal" if np.argmax(prediction[0]) == 0 else "Abnormal"
    confidence = float(prediction[0][np.argmax(prediction[0])])

    return result, confidence


model = create_model()

csv_path = "dataset/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv"
data = pd.read_csv(csv_path)

correct_predictions = 0
total_images = len(data)

for _, row in data.iterrows():
    image_name = row['X_ray_image_name']
    true_label = row['Label']
    image_path = os.path.join(
        "dataset/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/",
        image_name)
    print(image_path)

    if os.path.exists(image_path):
        predicted_label, _ = predict_xray(image_path)
        if predicted_label == true_label:
            correct_predictions += 1
        print(f"Image: {image_name}, True Label: {true_label}, Predicted Label: {predicted_label}")
    else:
        print(f"Image {image_name} not found.")

# Print overall accuracy
accuracy = correct_predictions / total_images
print(f"Accuracy: {accuracy:.2f}")

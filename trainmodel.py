# pip install kaggle
# pip install tensorflow
# pip install opencv-python
# pip install scikit-learn

import os
import zipfile
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def download_dataset():
    os.system("kaggle datasets download -d fournierp/captcha-version-2-images")
    with zipfile.ZipFile("captcha-version-2-images.zip", "r") as zip_ref:
        zip_ref.extractall("captchas")

def load_data(data_dir, img_size, max_label_length):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        label = os.path.splitext(filename)[0]
        if len(label) == max_label_length:
            label_indices = label_to_index(label)  # Convert the label to integer indices
            images.append(img)
            labels.append(label_indices)
    return np.array(images), np.array(labels)

def create_cnn_model(input_shape, num_classes, max_label_length):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = [Dense(num_classes, activation="softmax")(x) for _ in range(max_label_length)]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def label_to_index(label):
    index = []
    for char in label:
        if char.isdigit():
            index.append(int(char) + 26)  # Shift the digits to the index range [26, 35]
        else:
            index.append(ord(char.upper()) - ord("A"))  # Convert uppercase letters to the index range [0, 25]
    return np.array(index)

def index_to_label(index):
    label = ""
    for idx in index:
        if 0 <= idx <= 25:
            label += chr(idx + ord("A"))
        elif 26 <= idx <= 35:
            label += str(idx - 26)
    return label

def main():
    download_dataset()
    data_dir = "captchas/samples"
    img_size = (200, 50)
    input_shape = (img_size[1], img_size[0], 1)
    num_classes = 36  # 26 uppercase letters + 10 digits
    max_label_length = 5

    X, y = load_data(data_dir, img_size, max_label_length)
    X = X.reshape(-1, img_size[1], img_size[0], 1).astype("float32") / 255.0
    y = np.array(y)
    y = [to_categorical(y[:, i], num_classes=num_classes) for i in range(max_label_length)]

    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y).transpose(1, 0, 2), test_size=0.2, random_state=42)
    y_train = [y_train[:, i] for i in range(max_label_length)]
    y_test = [y_test[:, i] for i in range(max_label_length)]

    model = create_cnn_model(input_shape, num_classes, max_label_length)
    checkpoint = ModelCheckpoint("captcha_model.h5", monitor="val_loss", save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, callbacks=[checkpoint])

if __name__ == "__main__":
    main()



import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

def preprocess_image(image_path, img_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    img = cv2.resize(img, img_size)
    img = img.reshape(-1, img_size[1], img_size[0], 1).astype("float32") / 255.0
    return img

def index_to_label(index):
    if 0 <= index <= 25:
        return chr(index + ord("A"))
    elif 26 <= index <= 35:
        return str(index - 26)

def predict_captcha(model, img, index_to_label):
    predictions = model.predict(img)
    captcha = "".join(index_to_label(np.argmax(pred)) for pred in predictions)
    return captcha

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_captcha.py <path_to_captcha_image>")
        sys.exit(1)

    model_path = "captcha_model.h5"
    image_path = sys.argv[1]
    img_size = (200, 50)

    model = load_model(model_path)
    img = preprocess_image(image_path, img_size)
    if img is not None:
        captcha = predict_captcha(model, img, index_to_label)
        print(f"Predicted CAPTCHA: {captcha}")

if __name__ == "__main__":
    main()

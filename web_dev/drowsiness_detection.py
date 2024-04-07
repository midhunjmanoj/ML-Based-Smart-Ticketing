import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
from PIL import Image
import io


labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145


def prepare(img, face_cas="./haarcascade_frontalface_default.xml"):
    face_cascade = cv2.CascadeClassifier(face_cas)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    
    # If no faces are detected, return the resized image itself
    resized_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


def predict_drowsiness(image):
    model = tf.keras.models.load_model("Model/drowiness_new6.h5")
    prediction = model.predict([prepare(image)])
    print("Detected Label :", labels_new[np.argmax(prediction)])
    return labels_new[np.argmax(prediction)]



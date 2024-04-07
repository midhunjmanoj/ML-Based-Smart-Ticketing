import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
from PIL import Image
import io

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('Model/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img):
    # Convert PIL Image to NumPy array
    image = np.array(img)

    # Now you can access the shape attribute
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections


def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

def drawings(image, boxes_np, confidences_np, index):
    license_text = ''  # Default license text
    for ind in index.flatten():
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image, license_text  # Return the processed image and license text



def yolo_predictions(img):
    print("Chceking for the error hi: ",type(img), len(img))

    img = Image.open(io.BytesIO(img))
    img_np = np.array(img)  # Convert PIL Image to NumPy array
    input_image, detections = get_detections(img_np)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img, license_text = drawings(input_image, boxes_np, confidences_np, index)  # Pass input_image instead of img
    # Save or return the result image as needed
    return result_img, license_text

     
# extrating text
def extract_text(image,bbox):
    image = np.array(image)
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        print("Number plate found : ",text)
        return text

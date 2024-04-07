from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import os
import deeplearning as dl
from io import BytesIO
import drowsiness_detection as dd
import numpy as np

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(
    BASE_PATH, 'G:/Web Dev/predict/static/upload')

@app.route('/')
def home():
    return render_template('index.html', upload=False)

# @app.route('/drowsinessdetection', methods=['POST'])
# def drowsinessdetection():
#     if 'image' not in request.files:
#         return 'No image file provided', 400
#     image_file = request.files['image']
#     if image_file.filename == '':
#         return 'No selected image file', 400
#     image_data = image_file.read()
#     drowsiness_type = dd.predict_drowsiness(image_data)

#     return drowsiness_type

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/drowsinessdetection', methods=['POST'])
def drowsinessdetection():
    if 'image' not in request.files:
        return 'No image file provided', 400
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image file', 400
    image_data = image_file.read()
    image = Image.open(BytesIO(image_data))
    image_np = np.array(image)
    drowsiness_type = dd.predict_drowsiness(image_np)

    return drowsiness_type

@app.route('/processimage', methods=['POST'])
def processimage():
    print("received Image")
    if 'image' not in request.files:
        return 'No image file provided', 400

    image_file = request.files['image']
    if image_file.filename == '':
        return 'No selected image file', 400

    # Read image data from the file
    image_data = image_file.read()

    # Call the yolo_predictions function with the image data
    # Assuming yolo_predictions returns a tuple (processed_image, additional_string)
    processed_image, additional_string = dl.yolo_predictions(image_data)
    image = Image.fromarray(processed_image.astype('uint8'))  # Convert ndarray to PIL Image

    # Save the image to a bytes buffer
    buffer = BytesIO()
    image.save(buffer, format='PNG')  # You can change 'PNG' to another format if needed
    byte_data = buffer.getvalue()
    # Convert processed_image to base64 for JSON response
    processed_image_base64 = base64.b64encode(byte_data).decode('utf-8')

    response_data = {
        'image': processed_image_base64,
        'additional_string': additional_string
    }
    return jsonify(response_data)




if __name__ == "__main__":
    app.run(debug=True)
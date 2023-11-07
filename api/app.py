
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


import os
import base64
import io

app = Flask(__name)
model = keras.models.load_model('models/svm_gamma.joblib')
def predict(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0

    prediction = model.predict(np.expand_dims(image, axis=0))
    digit = np.argmax(prediction)

    return digit

@app.route('/predict_images', methods=['POST'])
def predict_images():
    try:
        data = request.get_json()
        
        if 'im1' in data and 'im2' in data:
            temp_image1 = Image.open(io.BytesIO(base64.b64decode(data['im1'])))
            temp_image2 = Image.open(io.BytesIO(base64.b64decode(data['im2'])))
            digit1 = predict(temp_image1)
            digit2 = predict(temp_image2)
            resp = digit1 == digit2

            return jsonify({'result': resp})

        else:
            return jsonify({'error': 'Invalid JSON data'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
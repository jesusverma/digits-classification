from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('models/svm_gamma.joblib')

def preprocess_image(image_bytes, size=(8, 8)):
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize(size, Image.LANCZOS)
    image_array = np.array(image).reshape(1, -1)
    return image_array

def predict_digit(image_array):
    return model.predict(image_array)[0]

def compare_digits(image1_bytes, image2_bytes):
    try:
        image1_arr = preprocess_image(image1_bytes)
        image2_arr = preprocess_image(image2_bytes)

        pred1 = predict_digit(image1_arr)
        pred2 = predict_digit(image2_arr)

        result = pred1 == pred2
        return bool(result)

    except Exception as e:
        return str(e)

@app.route('/predict_images', methods=['POST'])
def compare_digits_route():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify(error='Please provide two images.'), 400

        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        result = compare_digits(image1_bytes, image2_bytes)

        if isinstance(result, bool):
            return jsonify(same_digit=result)
        else:
            return jsonify(error=result)

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

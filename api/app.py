from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn.preprocessing import Normalizer
from io import BytesIO
import joblib
import os


app = Flask(__name__)

# Load the model
model = joblib.load('../models/m22aie203_svm.joblib')

model_dir = "../models"


normalizer = Normalizer(norm='l2')

models = {}


def load_model():
   models['tree'] = joblib.load(os.path.join(model_dir, 'm22aie203_tree.joblib'))
   models['svm'] = joblib.load(os.path.join(model_dir, 'm22aie203_svm.joblib'))
   models['logistic_regression'] = joblib.load(os.path.join(model_dir, 'm22aie203_logistic_regression.joblib'))
    

# Load models
load_model()

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


@app.route('/predict_accor_model/<model_type>', methods=['POST'])
def predict_accor_model(model_type):
    if model_type not in models:
        return jsonify(error='Model type not supported.'), 400

    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    model = models[model_type]

    image_bytes = request.files['image'].read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize((8, 8), Image.LANCZOS)
    
    image_arr = np.array(image).reshape(1, -1)
    image_arr_normalized = normalizer.transform(image_arr)
    pred = model.predict(image_arr_normalized)

    return jsonify(predicted_digit=int(pred[0]))

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

def process_image_bytes(image_bytes):
    processed_image = Image.open(BytesIO(image_bytes)).convert('L')
    processed_image = processed_image.resize((8, 8), Image.LANCZOS)
    processed_image_array = np.array(processed_image).reshape(1, -1)
    return processed_image_array

@app.route('/predict_image_of_digit', methods=['POST'])
def predict_digit():
    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    user_image_bytes = request.files['image'].read()

    processed_image_array = process_image_bytes(user_image_bytes)
    prediction_result = model.predict(processed_image_array)

    return jsonify(predicted_digit=int(prediction_result[0]))

if __name__ == '__main__':
    app.run(debug=True)

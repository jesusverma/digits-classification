import requests
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from io import BytesIO

# Load the digits dataset
digits_data = load_digits()

def send_prediction_request(image_bytes, model_type, base_url="http://0.0.0.0:5000/predict_accor_model/"):
    url = base_url + model_type
    image_bytes.seek(0)
    files = {'image': ('image.png', image_bytes, 'image/png')}
    response = requests.post(url, files=files)
    return response

# Select an image
chosen_index = 100  # Replace with your chosen index

# Convert the image to bytes
image_bytes = BytesIO()
plt.imsave(image_bytes, digits_data.images[chosen_index], cmap='gray', format='png')
image_bytes.seek(0)

# Model types
available_model_types = ['logistic_regression', 'tree', 'svm']

# Send the request for each model type
for model_type in available_model_types:
    response = send_prediction_request(image_bytes, model_type)
    if response.status_code == 200:
        print(f"Response from modal type {model_type} model:", response.json())
        print(f"Original label: {digits_data.target[chosen_index]}")
    else:
        print(f"Failed to get a response from the {model_type} model server:", response.status_code)

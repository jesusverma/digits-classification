from app import app
import pytest
from sklearn import datasets
from PIL import Image
from io import BytesIO

def convert_image_to_bytes(image):
    pil_image = Image.fromarray(image.astype('uint8'))
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def get_image_bytes_for_digit(digit, images, labels):
    index = next(i for i, label in enumerate(labels) if label == digit)
    return convert_image_to_bytes(images[index])

def send_predict_digit_request(image_bytes):
    return app.test_client().post(
        '/predict_image_of_digit', 
        data={'image': (BytesIO(image_bytes), 'image.png')},
        content_type='multipart/form-data'
    )

def assert_response_status(response, expected_status):
    assert response.status_code == expected_status
    print(f"Status Code Assertion Successful: {response.status_code}")

def assert_predicted_digit(response, expected_digit):
    assert response.get_json()['predicted_digit'] == expected_digit
    print(f"Verification successful for digit {expected_digit}")

def test_post_predict_digit():
    digits = datasets.load_digits()
    X, y = digits.images, digits.target

    for digit in range(10):
        print(f"Currently Processing Digit: {digit}")
        image_bytes = get_image_bytes_for_digit(digit, X, y)
        response = send_predict_digit_request(image_bytes)

        assert_response_status(response, 200)
        assert_predicted_digit(response, digit)

if __name__ == "__main__":
    pytest.main()

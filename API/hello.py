from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

# # Define the path to the 'models' directory
# models_directory = os.path.join(os.path.dirname(__file__), 'models')
# # Specify the name of the model file
# model_filename = 'best_model gamma:0.01_C:1.joblib'
# # Construct the full path to the model file
# model_path = os.path.join(models_directory, model_filename)
# # Load the model
# model = joblib.load(model_path)

# # Specify the path to the model file
# model_path = './models/best_model gamma:0.01_C:1.joblib'
# # # Load the model
# model = joblib.load(model_path)

model = '../best_model gamma:0.01_C:1.joblib'
# model = '/models/best_model gamma:0.01_C:1.joblib'
# Verify that the model is loaded successfully
if model:
    print("Model loaded successfully.")
else:
    print("Model loading failed.")

@app.route('/compare_image', methods=['POST'])
def compare_image():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify(error='Please provide two images.'), 400

    image1_bytes = request.files['image1'].read()
    image2_bytes = request.files['image2'].read()

    image1 = Image.open(BytesIO(image1_bytes)).convert('L')
    image2 = Image.open(BytesIO(image2_bytes)).convert('L')

    image1 = image1.resize((8, 8), Image.LANCZOS)
    image2 = image2.resize((8, 8), Image.LANCZOS)

    image1_arr = np.array(image1).reshape(1, -1)
    image2_arr = np.array(image2).reshape(1, -1)

    pred1 = model.predict(image1_arr)
    pred2 = model.predict(image2_arr)

    result = pred1 == pred2

    return jsonify(same_digit=bool(result[0]))

if __name__ == '__main__':
    app.run(debug=True)

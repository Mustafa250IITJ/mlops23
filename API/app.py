from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO

import joblib
import os

# model_folder = '../models'
# model = os.path.join(model_folder, 'best_model gamma:0.01_C:1.joblib')
# model = joblib.load('model.joblib')
model = joblib.load('best_model gamma:0.01_C:1.joblib')

app = Flask(__name__)

@app.route('/')
def hello():
    return 'MLops AQuiz-4'


@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    image_bytes = request.files['image'].read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = image.resize((8, 8), Image.LANCZOS)
    
    image_arr = np.array(image).reshape(1, -1)
    pred = model.predict(image_arr)

    return jsonify(predicted_digit=int(pred[0]))


@app.route('/compare_digits', methods=['POST'])
def compare_digits():
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



# #--------------asg-5-----------
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# import joblib
# import os
# # Folder where the trained machine learning model is saved
# model_folder = '../trainmodelsaved/'
# # model_folder = '/app/trainmodelsaved'
# # Construct the full path to the trained model
# model_path = os.path.join(model_folder, 'mnist_model.joblib')
# # Load the trained machine learning model
# loaded_model = joblib.load(model_path)

# # # Access the parameters/attributes of the model
# # if isinstance(loaded_model, LogisticRegression):
# #     # Assuming the model is a Logistic Regression model
# #     coefficients = loaded_model.coef_
# #     intercept = loaded_model.intercept_

# #     print("Coefficients:", coefficients)
# #     print("Intercept:", intercept)
# # else:
# #     print("Model type not recognized. Check the model type and access attributes accordingly.")


# @app.route('/')
# def hello():
#     return 'MLops Assignment-5'

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Assuming the input data is in JSON format
#         input_data = request.json

#         # Extract features from input_data 
#         # features = [input_data['feature1'], input_data['feature2'], ...]
#         # features = [input_data[f'feature{i}'] for i in range(1, 65)]
#         features = [input_data.get(f'feature{i}', 0) for i in range(1, 65)]

#         # Perform prediction using the loaded model
#         prediction = loaded_model.predict([features])

#         # Convert the prediction to a dictionary
#         result = {"prediction": int(prediction[0])}

#         return jsonify(result)

#     except Exception as e:
#         # Handle exceptions appropriately
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run()
#     # app.run(host='0.0.0.0', port=80)
##---------end


###==================--------notes---------==============

# #------- run flask
# export FLASK_APP=app         # To run the application, use the flask command or python -m flask. Before you can do that you need to tell your
                                 # terminal the application to work with by exporting the FLASK_APP environment variable:
# export FLASK_ENV=development   # To enable all development features, set the FLASK_ENV environment variable to development before calling flask run.
# export FLASK_APP=app:app     # To run the application, use the flask command or python -m flask. Before you can do that you need to tell your
                                 # terminal the application to work with by exporting the FLASK_APP environment variable:
# flask run
# # ---------

# #------- run flask in docker container
# docker build -t predict:v1 -f Docker/dockerfile .
# docker run -it -p 5000:5000 predict:v1 bash
##---- individual run the flask file
# # change the directory where flask code is present (cd API) 
# export FLASK_APP=app
# flask run --host=0.0.0.0
# # ---------


# # -----open your web browser and navigate to 
# # http://127.0.0.1:5000
# # # # ---------------or------------
# # run the file using "flask run" or "python <filename>.py", then open another terminal then type
# # curl "http://127.0.0.1:5000     # to be chacked


# # -----To make a POST request, you can use a tool like curl. For example
# # run the file using "flask run" or "python <filename>.py", then open another terminal then type
# # curl -X POST -H "Content-Type: application/json" -d '{"key1": "value1", "key2": "value2"}' http://127.0.0.1:5000/predict
# # curl -X POST -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, ...}' http://127.0.0.1:5000/predict
# # curl -X POST -H "Content-Type: application/json" -d '{"feature1": 5, "feature2": 10}' http://127.0.0.1:5000/predict
# # curl http://127.0.0.1:5000/predict -X POST -H "Content-Type: application/json" -d '{"feature1": 5, "feature2": 10}'
# # curl http://127.0.0.1:5000/predict -X POST -H "Content-Type: application/json" -d '{"feature1": 5,"feature2": 10,"feature3": 0,"feature4": 0,"feature5": 0, "feature6": 0, "feature7": 0, "feature8": 0, "feature9": 0, "feature10": 0, "feature11": 0, "feature12": 0, "feature13": 0, "feature14": 0, "feature15": 0, "feature16": 0, "feature17": 0, "feature18": 0, "feature19": 0, "feature20": 0, "feature21": 0, "feature22": 0, "feature23": 0, "feature24": 0, "feature25": 0, "feature26": 0, "feature27": 0, "feature28": 0, "feature29": 0, "feature30": 0, "feature31": 0, "feature32": 0, "feature33": 0, "feature34": 0, "feature35": 0, "feature36": 0, "feature37": 0, "feature38": 0, "feature39": 0, "feature40": 0, "feature41": 0, "feature42": 0, "feature43": 0, "feature44": 0, "feature45": 0, "feature46": 0, "feature47": 0, "feature48": 0, "feature49": 0, "feature50": 0, "feature51": 0, "feature52": 0, "feature53": 0, "feature54": 0, "feature55": 0, "feature56": 0, "feature57": 0, "feature58": 0, "feature59": 0, "feature60": 0, "feature61": 0, "feature62": 0, "feature63": 0, "feature64": 0}' 
# # ---------------------

##----------sample code
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return 'Hello, World!'

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Assuming the input data is in JSON format
#         input_data = request.json

#         # Perform prediction or processing based on your model or application logic
#         # Replace this with your actual prediction logic
#         result = {"prediction": "This is a sample prediction."}

#         return jsonify(result)

#     except Exception as e:
#         # Handle exceptions appropriately
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run()
#     # app.run(host='0.0.0.0', port=80)
##------
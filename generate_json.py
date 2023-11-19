import json
import joblib
import numpy as np
import os

# Folder where the trained machine learning model is saved
model_folder = 'trainmodelsaved/'
# model_folder = '/app/trainmodelsaved'
# Construct the full path to the trained model
model_path = os.path.join(model_folder, 'mnist_model.joblib')
# Load the trained machine learning model
loaded_model = joblib.load(model_path)

# Create sample data for testing (adjust the values based on your model requirements)
sample_data = np.zeros((1, 64))  # Assuming the model expects an input with 64 features

# Make a prediction
prediction = loaded_model.predict(sample_data)

# Convert the prediction and sample data to a dictionary
result = {
    "sample_data": sample_data.tolist(),
    "prediction": int(prediction[0])
}

# Save the result as a JSON file
with open('model_prediction.json', 'w') as json_file:
    json.dump(result, json_file, indent=2)

print("JSON file 'model_prediction.json' has been generated.")

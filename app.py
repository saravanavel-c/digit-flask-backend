from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load the trained model once when server starts
model = load_model("mnist_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    base64_str = data['image'].split(",")[1]  # remove header
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # convert to grayscale
    img = img.resize((28, 28))  # resize to 28x28

    # Preprocess for model input
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({"prediction": predicted_digit})

if __name__ == "__main__":
    app.run(debug=True)

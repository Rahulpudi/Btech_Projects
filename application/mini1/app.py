from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "combinednetb7.keras"
model = load_model(MODEL_PATH)

classes = [
    "Alzheimer_MildDemented",
    "Alzheimer_ModerateDemented",
    "Alzheimer_VeryMildDemented",
    "Alzheimer_NonDemented",
    "BrainTumor_Glioma",
    "BrainTumor_Meningioma",
    "BrainTumor_Normal",
    "BrainTumor_Pitutary;"
]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ✅ Home route
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    preds = model.predict(img_array)
    pred_class = classes[np.argmax(preds)]
    confidence = float(np.max(preds))

    os.remove(file_path)

    return jsonify({
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "all_probabilities": {classes[i]: float(preds[0][i]) for i in range(len(classes))}
    })

if __name__ == "__main__":
    app.run(debug=True)

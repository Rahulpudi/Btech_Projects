# app.py
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import warnings
warnings.filterwarnings('ignore')

# =========================
# 1. Flask App Initialization
# =========================
app = Flask(__name__)

# =========================
# 2. Model Configuration
# =========================
MODEL_PATH = "EfficientNetB3_combined_dataset.keras"
IMG_SIZE = (244, 244)

# Class names from your training code
class_names = [
    "Control_Axial", "Control_Saggital", "Glioma_Brain_Tumor", "Multiple_Sclerosis_Axial",
    "Multiple_Sclerosis_Saggital", "Meningioma_Brain_Tumor", "MildDemented_Alzheimer", "ModerateDemented_Alzheimer",
    "NonDemented", "Normal", "Pituitary_Brain_tumor", "VeryMildDemented_Alzheimer"
]
# =========================
# 3. Build Model Architecture
# =========================
def build_model():
    base_model = EfficientNetB3(
        include_top=False, 
        weights=None,
        input_shape=(*IMG_SIZE, 3), 
        pooling="max"
    )
    
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == "block6a_expand_conv":
            set_trainable = True
        layer.trainable = set_trainable

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(len(class_names), activation="softmax")
    ])
    
    return model

# Build the model
model = build_model()

# Try to load weights
try:
    model = load_model(MODEL_PATH)
    print("Full model loaded successfully")
except:
    try:
        model.load_weights(MODEL_PATH)
        print("Weights loaded successfully")
    except:
        try:
            weights_path = MODEL_PATH.replace('.keras', '.h5')
            model.load_weights(weights_path)
            print("Weights loaded from .h5 file")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using randomly initialized weights")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================
# 4. Helper Functions
# =========================
def preprocess_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_fn(images):
    """Wrapper function for model prediction that LIME expects"""
    # Preprocess the images for EfficientNet
    processed_imgs = preprocess_input(images.astype('float32'))
    return model.predict(processed_imgs)

def generate_lime_explanation(img_path, top_labels=5, num_features=10, num_samples=50):
    """Generate LIME explanation for the image"""
    # Load and preprocess the image
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Normalize for LIME
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Explain the image
    explanation = explainer.explain_instance(
        img_array, 
        predict_fn, 
        top_labels=top_labels, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    # Get the top label from the explanation
    top_label = explanation.top_labels[0]
    
    # Get explanation for the top label
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Create the explanation visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Encode the image to base64
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return encoded_img, top_label

# =========================
# 5. Routes
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_img(file_path)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = float(np.max(preds))

        # Generate LIME explanation - use the top label from LIME
        lime_img, lime_top_label = generate_lime_explanation(file_path)
        lime_pred_class = class_names[lime_top_label]

        # Get top predictions
        top_indices = np.argsort(preds[0])[-5:][::-1]
        top_predictions = [
            {"class": class_names[i], "confidence": float(preds[0][i])}
            for i in top_indices
        ]

        os.remove(file_path)

        return jsonify({
            "predicted_class": pred_class,
            "lime_predicted_class": lime_pred_class,  # Add LIME's prediction
            "confidence": confidence,
            "lime_explanation": lime_img,
            "top_predictions": top_predictions
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {error_details}")
        return jsonify({"error": str(e)}), 500

# =========================
# 6. Run Flask
# =========================
if __name__ == '__main__':
    app.run(debug=True)
import os
import uuid
import numpy as np
from flask import Flask, render_template, request, url_for
from PIL import Image

from utils import (
    load_all_models,
    allowed_file,
    predict_image,
    create_gradcam_explanation,
    get_disease_info,
)
from gradcam import generate_gradcam_from_tensor

BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"]     = GRADCAM_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Load all models once at startup
bundle = load_all_models(BASE_DIR)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No image file was uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please choose an image first.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload a JPG, JPEG, or PNG image."
        )

    # Save the uploaded file
    ext         = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    image_path  = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(image_path)

    try:
        # ── Predict ──────────────────────────────────────────────────────────
        result = predict_image(image_path, bundle)

        # Use the original unmasked image as the base for the overlay
        original_np = result["original_np"]   # RGB uint8 array at 224×224

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        gradcam_name = f"gradcam_{uuid.uuid4().hex}.jpg"
        gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_name)

        generate_gradcam_from_tensor(
            input_tensor   = result["input_tensor"],
            original_image = original_np,
            output_path    = gradcam_path,
            model          = bundle["full_model"],
            target_layer   = bundle["target_layer"],
            class_index    = result["pred_index"],
            device         = bundle["device"],
        )

        # ── Build context for the template ───────────────────────────────────
        explanation  = create_gradcam_explanation(
            result["predicted_label"], result["confidence"]
        )
        disease_info = get_disease_info(result["predicted_label"])

        return render_template(
            "result.html",
            original_image  = url_for("static", filename=f"uploads/{unique_name}"),
            gradcam_image   = url_for("static", filename=f"gradcam/{gradcam_name}"),
            predicted_label = result["predicted_label"],
            confidence      = f"{result['confidence'] * 100:.2f}",
            top_predictions = result["top_predictions"],
            explanation     = explanation,
            disease_info    = disease_info,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
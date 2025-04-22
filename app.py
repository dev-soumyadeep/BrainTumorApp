import os
import traceback
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from datetime import datetime

app = Flask(__name__)

# Load binary classification models
model1 = load_model('best_model1.h5')
model2 = load_model('best_model2.h5')

# Load multimodal classification models
classification_models = []
model_paths = [
    'best_model3.h5',
    'best_model4.h5',
    'best_model5.h5',
    'best_model6.h5',
    'best_model7.h5',
    'best_model8.h5'
]
for path in model_paths:
    try:
        m = load_model(path)
        classification_models.append(m)
        print(f"{path} loaded successfully.")
    except Exception as e:
        print(f"Error loading {path}: {e}")

# Tumor label mapping
tumor_labels = ['Glioma', 'Meningioma', 'Pituitary', 'Other']

# Tumor info lookups
tumor_descriptions = {
    'Glioma':     "Gliomas are tumors that arise from glial cells in the brain and spine. They can be slow or fast growing.",
    'Meningioma': "Meningiomas develop from the meninges, the membranes that surround your brain and spinal cord. Typically benign.",
    'Pituitary':  "Pituitary tumors occur in the pituitary gland and can affect hormone production.",
    'Other':      "Other less common or unspecified tumor types."
}

tumor_medications = {
    'Glioma':     ["Temozolomide", "Bevacizumab"],
    'Meningioma': ["Dexamethasone", "Hydroxyurea"],
    'Pituitary':  ["Cabergoline", "Pegvisomant"],
    'Other':      ["Consult specialist"]
}

# Upload config
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if missing
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path, target_size):
    """Reads and preprocesses image to match the target size."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=filename))
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        input_shape = model1.input_shape
        target_size = (input_shape[1], input_shape[2])
        img = preprocess_image(img_path, target_size)
        pred1 = model1.predict(img)[0][0]
        pred2 = model2.predict(img)[0][0]
        class1 = 1 if pred1 >= 0.5 else 0
        class2 = 1 if pred2 >= 0.5 else 0
        show_classification_button = False
        if class1 == 1 and class2 == 1:
            final_prediction = "Tumor Detected"
            tumor_detected = True
            show_classification_button = True
        elif class1 == 0 and class2 == 0:
            final_prediction = "No Tumor Detected"
            tumor_detected = False
        else:
            combined_prob = (pred1 + pred2) / 2.0
            tumor_detected = False
            if combined_prob > 0.4:
                final_prediction = (
                    "Models disagree.<br>"
                    f"Model1 Probability: {pred1:.2f}<br>"
                    f"Model2 Probability: {pred2:.2f}<br>"
                    f"Combined Probability: {combined_prob:.2f}<br>"
                    "Tumor May Exist"
                )
                show_classification_button = True
            else:
                final_prediction = (
                    "Models disagree.<br>"
                    f"Model1 Probability: {pred1:.2f}<br>"
                    f"Model2 Probability: {pred2:.2f}<br>"
                    f"Combined Probability: {combined_prob:.2f}"
                )
    except Exception as e:
        return f"Error during binary classification: {e}"

    return render_template(
        'prediction.html',
        prediction=final_prediction,
        filename=filename,
        tumor_detected=tumor_detected,
        show_classification_button=show_classification_button
    )

@app.route('/classify/<filename>')
def classify(filename):
    if not classification_models:
        return "No classification models available."
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            raise ValueError("Image not found or unreadable.")
        preds = []
        for m in classification_models:
            h, w = m.input_shape[1], m.input_shape[2]
            img = cv2.resize(img_raw, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            p = m.predict(img)
            idx = int(np.argmax(p))
            preds.append(tumor_labels[idx])
        final_label = max(set(preds), key=preds.count)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return f"<h3>Classification Error</h3><pre>{e}</pre>"

    # Lookup description + meds
    desc = tumor_descriptions.get(final_label, "No description available.")
    meds = tumor_medications.get(final_label, [])

    return render_template(
        'classification.html',
        filename=filename,
        tumor_type=final_label,
        tumor_description=desc,
        medications=meds
    )

@app.route('/download_report/<filename>')
def download_report(filename):
    try:
        # Get image path
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Re-run classification to get tumor type
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            return "Unable to read image for report."

        predictions = []
        for model in classification_models:
            input_shape = model.input_shape
            height, width = input_shape[1], input_shape[2]
            img = cv2.resize(img_raw, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            label_index = int(np.argmax(pred))
            predictions.append(tumor_labels[label_index])

        final_label = max(set(predictions), key=predictions.count)
        description = tumor_descriptions.get(final_label, "N/A")
        meds = tumor_medications.get(final_label, ["N/A"])

        # Create PDF in memory
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, height - 50, "Brain Tumor Detection Report")

        # Metadata
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 80, f"Filename: {filename}")
        p.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Tumor Type
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 140, f"Tumor Type: {final_label}")

        # Description
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 170, "Description:")
        text_object = p.beginText(60, height - 190)
        for line in description.split('. '):
            text_object.textLine(line.strip())
        p.drawText(text_object)

        # Medications
        p.drawString(50, height - 330, "Recommended Medications:")
        y_pos = height - 350
        for med in meds:
            p.drawString(70, y_pos, f"- {med}")
            y_pos -= 20

        # Finalize PDF
        p.showPage()
        p.save()

        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="tumor_report.pdf", mimetype='application/pdf')

    except Exception as e:
        return f"Failed to generate report: {e}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    # app.run(host='0.0.0.0', port=3000)



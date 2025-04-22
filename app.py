import os
import traceback
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
from datetime import datetime
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet

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
    'Glioma':     "Gliomas are tumors that arise from glial cells in the brain and spine. They can be slow or fast growing.Glioma is a common type of tumor originating in the brain, but it can sometimes be found in the spinal cord. About 33% of all brain tumors are gliomas. These tumors arise from the glial cells that surround and support neurons. There are several types of glial cells, hence there are many types of gliomas, including: astrocytomas, oligodendrogliomas, and ependymomas. Gliomas can be classified as low-grade (slow-growing) or high-grade (fast-growing). High-grade gliomas are more aggressive and can be life-threatening. The most common type of glioma is glioblastoma multiforme (GBM), which is a high-grade tumor. Gliomas can occur at any age but are more common in adults.The understanding of gliomas has been evolving over the years. Depending on the type of cells that are forming the glioma and their genetic mutations, those tumors can be more or less aggressive. A genetic study of the tumor is often performed to better understand how it may behave. For example, diffuse midline gliomas or hemispheric gliomas are newly described types of gliomas that have specific mutations associated with a more aggressive nature. ",
    
    'Meningioma': "Meningiomas develop from the meninges, the membranes that surround your brain and spinal cord. Typically benign.Meningioma is the most common primary brain tumor, accounting for more than 30% of all brain tumors. Meningiomas originate in the meninges, the outer three layers of tissue that cover and protect the brain just under the skull. Women are diagnosed with meningiomas more often than men. About 85% of meningiomas are noncancerous, slow-growing tumors. Almost all meningiomas are considered benign, but some meningiomas can be persistent and come back after treatment.",
    
    'Pituitary':  "Pituitary tumors occur in the pituitary gland and can affect hormone production.Adenoma, a type of tumor that grows in the gland tissues, is the most common type of pituitary tumor. Pituitary adenomas develop from the pituitary gland and tend to grow at a slow rate. About 10% of primary brain tumors are diagnosed as adenomas. They can cause vision and endocrinological problems. Fortunately for patients affected by them, adenomas are benign and treatable with surgery and/or medication.",
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
        # Image path
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Read and preprocess image
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
        pdf_width, pdf_height = letter

        # Header
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, pdf_height - 50, "Brain Tumor Detection Report")

        # Metadata
        p.setFont("Helvetica", 12)
        p.drawString(50, pdf_height - 80, f"Filename: {filename}")
        p.drawString(50, pdf_height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Tumor Type
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, pdf_height - 140, f"Tumor Type: {final_label}")

        # MRI Image in PDF
        image_reader = ImageReader(img_path)
        image_width = 200
        image_height = 200
        p.drawImage(image_reader, pdf_width - image_width - 50, pdf_height - image_height - 140, 
                    width=image_width, height=image_height)

       # Set label for description
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, pdf_height - 360, "Description:")

        # Prepare paragraph style
        styles = getSampleStyleSheet()
        desc_para = Paragraph(description, styles["Normal"])

        # Create a frame to render the paragraph inside the page margins
        desc_frame = Frame(
            50,                    # X position
            pdf_height - 520,      # Y position (top-down)
            pdf_width - 100,       # Width (full width minus side margins)
            140                    # Height of the description box
        )

        # Draw the paragraph inside the frame
        desc_frame.addFromList([desc_para], p)

        # Medications
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, pdf_height - 550, "Recommended Medications:")

        p.setFont("Helvetica", 11)
        y_med = pdf_height - 570
        for med in meds:
            p.drawString(70, y_med, f"- {med}")
            y_med -= 18


        # Get Well Soon WordArt-like message
        p.setFont("Helvetica-BoldOblique", 18)
        p.setFillColorRGB(0.8, 0.1, 0.5)  # pinkish-purple
        p.drawCentredString(pdf_width / 2, 80, "💖 Get Well Soon! 💖")


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



import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model

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

# Upload config
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if missing
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size):
    """Reads and preprocesses image to match the target size (used per model)."""
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
        # Use model1's input size for preprocessing
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
    if len(classification_models) == 0:
        return "No classification models available."

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            raise ValueError("Image not found or unreadable.")

        predictions = []

        for model in classification_models:
            # Dynamically extract input shape
            input_shape = model.input_shape
            height, width = input_shape[1], input_shape[2]

            img = cv2.resize(img_raw, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            label_index = int(np.argmax(pred))
            predictions.append(tumor_labels[label_index])

        # Majority voting
        final_label = max(set(predictions), key=predictions.count)

    except Exception as e:
        return f"An error occurred during further classification: {e}"

    return render_template('classification.html', filename=filename, tumor_type=final_label)

if __name__ == "__main__":
    # app.run(debug=True, port=5000)
    app.run(host='0.0.0.0', port=3000)



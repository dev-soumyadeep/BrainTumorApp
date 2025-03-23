import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load both trained models
model1 = load_model('best_model1.h5')  # Adjust path if needed
model2 = load_model('best_model2.h5')  # Adjust path if needed

# Configure upload folder (inside static)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Home route for uploading images."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
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
    """
    Prediction route:
    - Reads the uploaded image
    - Preprocesses it
    - Gets predictions from both models
    - Applies the logic to combine results
    - Renders the prediction page
    """
    # Build the image path in static/uploads
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path)

    if img is None:
        return "Error loading image."

    # Preprocess the image
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predictions
    pred1 = model1.predict(img)[0][0]  # Probability from model1
    pred2 = model2.predict(img)[0][0]  # Probability from model2

    # Classify based on threshold 0.5
    class1 = 1 if pred1 >= 0.5 else 0
    class2 = 1 if pred2 >= 0.5 else 0

    # Combine results
    if class1 == 1 and class2 == 1:
        final_prediction = "Tumor Detected"
    elif class1 == 0 and class2 == 0:
        final_prediction = "No Tumor Detected"
    else:
        combined_prob = (pred1 + pred2) / 2.0
        final_prediction = (
            "Models disagree.<br>"
            f"Model1 Probability: {pred1:.2f}<br>"
            f"Model2 Probability: {pred2:.2f}<br>"
            f"Combined Probability: {combined_prob:.2f}"
        )

    # We only pass the raw filename to the template,
    # then the template references it with "uploads/filename"
    return render_template(
        'prediction.html',
        prediction=final_prediction,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)

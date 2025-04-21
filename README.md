# 🧠 Brain Tumor Detection and Classification Web App

This is a Flask-based web application that allows users to upload MRI images for brain tumor **detection** and **classification**. The app uses deep learning models to analyze MRI scans and determine whether a tumor is present and, if so, what type of tumor it is.

It also features a hospital locator to find nearby hospitals based on your current location.

---

## 🚀 Features

- 📤 Upload MRI brain scan images  
- 🔍 Detect presence of a tumor using a binary classifier  
- 🧠 Classify the tumor type: **Glioma**, **Meningioma**, **Pituitary**, or **Others**  
- 📍 Find nearby hospitals using geolocation and Overpass API  
- 💡 Clean, responsive UI built with **Bootstrap**

---

## 📸 Sample Screens

> Upload MRI image → View results → See nearby hospitals  
![Sample](static/sample-ui.png)  
*(Add your own screenshots here if available)*

---

## 🛠️ Setup Instructions

Follow these steps to run the project locally.

### 1. Clone the Repository and Enter the Directory

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```
### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```
### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Run the Web App Locally
```bash
python wsgi.py
```

### 5. Then open your browser and go to:
http://127.0.0.1:5000/

# Use an official slim Python runtime as a base image
FROM python:3.11-slim

# Install system dependencies, including Git and Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first for caching dependencies
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Pull Git LFS files (if needed)
RUN git lfs pull

# Expose the port your Flask app will run on
EXPOSE 5000

# Command to run your application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]


RUN pip freeze | grep gunicorn
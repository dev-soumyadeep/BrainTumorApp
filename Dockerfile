# Use an official slim Python runtime as a base image
FROM python:3.11-slim

# Install system dependencies, including Git LFS
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first, so Docker can cache the pip install layer if requirements don't change
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Expose the port that your Flask app uses
EXPOSE 5000

# Command to run your application (adjust if your app entry point is different)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

# Dockerfile

# Use a slim, stable Python base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (None needed for this app, but good practice)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ... \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
# Use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run expects the container to listen on the PORT environment variable (default 8080)
EXPOSE 8080

# The CMD defines the command to run the application using Gunicorn (a production WSGI server)
# The format is: gunicorn -b :$PORT <app_file>:<app_instance>
CMD ["gunicorn", "--time-out", "0", "--bind", "0.0.0.0:8080", "app:app"]

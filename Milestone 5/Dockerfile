# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install git for pip to be able to install from GitHub
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.lock .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.lock

# Copy the rest of the app
COPY . .

# Set default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

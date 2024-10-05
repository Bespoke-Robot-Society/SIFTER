# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
#COPY requirements.txt /app/
COPY . /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 9123

# Command to run the Flask app
CMD ["python", "hook.py"]

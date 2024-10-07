# Base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
#COPY requirements.txt /app/
COPY . /app/
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose the Dash app port
EXPOSE 8050 

# Command to run the Dash app
CMD ["python", "frontend/app.py"]

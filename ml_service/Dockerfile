# Use the official Python 3.11 slim image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 8080
EXPOSE 8080

# Set environment variable
ENV PORT=8080

# Run the FastAPI app with uvicorn on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

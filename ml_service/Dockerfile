# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Install OS-level dependencies if needed (e.g., gcc for compiling any package extensions)
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire application code for the ML service
COPY . .

# Expose the port used by the FastAPI app (uvicorn default: 8000)
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

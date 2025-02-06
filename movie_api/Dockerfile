# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Install OS-level dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends gcc

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Set environment variable (Cloud Run automatically sets PORT, but this is explicit)
ENV PORT=8080

# Run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8080"]

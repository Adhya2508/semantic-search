FROM python:3.10-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure python logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (important for large wheels like numpy)
RUN pip install --upgrade pip

# Copy the requirements first (better caching)
COPY requirements.txt .

# Install python dependencies with retries and long timeout
RUN pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries=10 \
    -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.py .
COPY main.py .

# Create directories for mounted data
RUN mkdir -p models data

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

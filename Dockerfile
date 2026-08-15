FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install torchvision --no-deps && \
    pip install -r requirements.txt

# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

EXPOSE 8501

# Run Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Base image with Python and system dependencies
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Ensure unstructured dependencies work
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip
RUN pip install "unstructured[all-docs]" pillow lxml
RUN pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Streamlit configuration (optional)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRC=false

# Run the Streamlit app
CMD ["streamlit", "run", "frontend/display.py"]

FROM ubuntu:22.04

WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install latest Python build tools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel scikit-build-core cmake ninja

# Set environment variables for CPU build
ENV LLAMA_CUBLAS=0
ENV SKBUILD_CUPY_PIP_INSTALL=0
ENV CMAKE_ARGS="-DLLAMA_BLAS=OFF -DBUILD_SHARED_LIBS=ON"
ENV PIP_FIND_LINKS="https://abetlen.github.io/llama-cpp-python/whl/cpu/"

# Install llama-cpp-python with explicit build requirements
RUN pip install --no-cache-dir \
    --verbose \ 
    --force-reinstall \
    "llama-cpp-python==0.2.85"

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

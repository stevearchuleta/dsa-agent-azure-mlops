# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package itself (no deps, already installed)
RUN pip install --no-cache-dir --no-deps -e .

# Create data and artifact directories
RUN mkdir -p data/papers data/faiss_index artifacts

# Default command: show help / version
CMD ["python", "-c", "import dsa; print('DSA Agent ready')"]
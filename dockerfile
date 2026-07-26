# Build stage
FROM python:3.13-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv for faster dependency management
RUN pip install uv

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies only (no dev dependencies)
RUN uv pip install --no-cache-dir \
    -e . \
    && uv pip install --no-cache-dir dvc

# Final stage
FROM python:3.13-slim AS runner

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY . .

# Create necessary directories (DVC will create outputs)
RUN mkdir -p data/raw data/processed data/features models logs metrics

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=mlruns
ENV PYTHONUNBUFFERED=1

# Create a script to run the pipeline
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting DVC pipeline..."\n\
\n\
# Check if raw data exists, if not, download it\n\
if [ ! -f "data/raw/ratings.csv" ] || [ ! -f "data/raw/movies.csv" ]; then\n\
    echo "Raw data not found. Attempting to pull from DVC remote..."\n\
    dvc pull || echo "No DVC remote configured. Please ensure raw data is mounted or downloaded."\n\
fi\n\
\n\
# Run the DVC pipeline\n\
if [ "$1" = "repro" ]; then\n\
    dvc repro\n\
elif [ "$1" = "pull" ]; then\n\
    dvc pull\n\
    echo "Data pulled successfully!"\n\
elif [ "$1" = "status" ]; then\n\
    dvc status\n\
else\n\
    echo "Running default: dvc repro"\n\
    dvc repro\n\
fi\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Create entrypoint script for better container lifecycle management
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
if [ "$1" = "pipeline" ]; then\n\
    exec /app/run_pipeline.sh "$2"\n\
elif [ "$1" = "shell" ]; then\n\
    exec /bin/bash\n\
elif [ "$1" = "test" ]; then\n\
    exec pytest tests/ -v\n\
elif [ "$1" = "mlflow" ]; then\n\
    exec mlflow ui --host 0.0.0.0 --port 5000\n\
else\n\
    exec "$@"\n\
fi\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - run the DVC pipeline
CMD ["pipeline", "repro"]
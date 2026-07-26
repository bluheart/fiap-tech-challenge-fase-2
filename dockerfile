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

# Install runtime system dependencies including git for DVC
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files (including .git for DVC)
COPY . .

# Initialize git repository if not present (for DVC)
RUN if [ ! -d ".git" ]; then \
        git init && \
        git add . && \
        git config user.email "container@local" && \
        git config user.name "Container User" && \
        git commit -m "Initial commit for DVC" || true; \
    fi

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features models logs metrics

# Set environment variables
ENV MLFLOW_TRACKING_URI=mlruns
ENV PYTHONUNBUFFERED=1
ENV DVC_NO_ANALYTICS=true

# Create a comprehensive pipeline runner script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting DVC pipeline..."\n\
echo "Current directory: $(pwd)"\n\
echo "Files in current directory:"\n\
ls -la\n\
\n\
# Check if DVC is initialized\n\
if [ ! -f ".dvc/config" ]; then\n\
    echo "Initializing DVC..."\n\
    dvc init --no-scm --quiet\n\
fi\n\
\n\
# Check if .dvc files exist for raw data\n\
if [ ! -f "data/raw/movies.csv.dvc" ] || [ ! -f "data/raw/ratings.csv.dvc" ]; then\n\
    echo "DVC files not found. Creating them..."\n\
    # Create DVC files for raw data if they exist\n\
    if [ -f "data/raw/movies.csv" ]; then\n\
        dvc add data/raw/movies.csv --no-commit\n\
    fi\n\
    if [ -f "data/raw/ratings.csv" ]; then\n\
        dvc add data/raw/ratings.csv --no-commit\n\
    fi\n\
fi\n\
\n\
# Handle different commands\n\
case "$1" in\n\
    repro)\n\
        echo "Running: dvc repro"\n\
        dvc repro --force\n\
        ;;\n\
    pull)\n\
        echo "Running: dvc pull"\n\
        dvc pull\n\
        ;;\n\
    status)\n\
        echo "Running: dvc status"\n\
        dvc status\n\
        ;;\n\
    remote)\n\
        echo "Setting up DVC remote..."\n\
        dvc remote add -d local ./storage\n\
        mkdir -p storage\n\
        ;;\n\
    list)\n\
        echo "Listing DVC tracked files:"\n\
        dvc list . --dvc-only\n\
        ;;\n\
    *)\n\
        echo "Running default: dvc repro"\n\
        dvc repro --force\n\
        ;;\n\
esac\n\
\n\
# Display results\n\
echo "Pipeline execution completed!"\n\
echo "Output files:"\n\
find data/processed data/features models -type f 2>/dev/null || echo "No output files found"\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
case "$1" in\n\
    pipeline)\n\
        exec /app/run_pipeline.sh "$2"\n\
        ;;\n\
    shell)\n\
        exec /bin/bash\n\
        ;;\n\
    test)\n\
        exec pytest tests/ -v\n\
        ;;\n\
    mlflow)\n\
        exec mlflow ui --host 0.0.0.0 --port 5000\n\
        ;;\n\
    bash)\n\
        exec /bin/bash\n\
        ;;\n\
    *)\n\
        exec "$@"\n\
        ;;\n\
esac\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["pipeline", "repro"]
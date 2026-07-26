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

# Copy project files
COPY . .

# Initialize git repository if not present (for DVC)
RUN if [ ! -d ".git" ]; then \
        git init && \
        git config user.email "container@local" && \
        git config user.name "Container User" && \
        echo "Initializing Git repository..."; \
    fi

# Create .gitignore to exclude data files from Git
RUN echo "# Data files (tracked by DVC)\n\
data/raw/*.csv\n\
data/processed/*.csv\n\
data/features/*.csv\n\
!data/raw/*.dvc\n\
\n\
# Models (tracked by DVC)\n\
models/*.pkl\n\
\n\
# MLflow\n\
mlruns/\n\
mlflow.db\n\
\n\
# Logs\n\
logs/\n\
*.log\n\
" > .gitignore

# Fix DVC setup - remove files from Git and add to DVC
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Setting up DVC in container..."\n\
\n\
# Initialize DVC without SCM integration (--no-scm) since we want to use it with Git\n\
if [ ! -d ".dvc" ]; then\n\
    dvc init --quiet\n\
fi\n\
\n\
# Remove CSV files from Git tracking if they exist\n\
if [ -f "data/raw/ratings.csv" ] && git ls-files --error-unmatch data/raw/ratings.csv 2>/dev/null; then\n\
    echo "Removing data/raw/ratings.csv from Git..."\n\
    git rm --cached data/raw/ratings.csv\n\
fi\n\
\n\
if [ -f "data/raw/movies.csv" ] && git ls-files --error-unmatch data/raw/movies.csv 2>/dev/null; then\n\
    echo "Removing data/raw/movies.csv from Git..."\n\
    git rm --cached data/raw/movies.csv\n\
fi\n\
\n\
# Add .gitignore to Git\n\
git add .gitignore\n\
\n\
# Commit changes\n\
git add . 2>/dev/null || true\n\
git commit -m "Initial setup for DVC" 2>/dev/null || echo "No changes to commit"\n\
\n\
# Create DVC files for raw data if they don'\''t exist and CSV files are present\n\
if [ -f "data/raw/ratings.csv" ] && [ ! -f "data/raw/ratings.csv.dvc" ]; then\n\
    echo "Adding ratings.csv to DVC..."\n\
    dvc add data/raw/ratings.csv --no-commit\n\
fi\n\
\n\
if [ -f "data/raw/movies.csv" ] && [ ! -f "data/raw/movies.csv.dvc" ]; then\n\
    echo "Adding movies.csv to DVC..."\n\
    dvc add data/raw/movies.csv --no-commit\n\
fi\n\
\n\
echo "DVC setup complete!"\n\
' > /app/setup_dvc.sh && chmod +x /app/setup_dvc.sh

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
echo "Starting DVC pipeline in container..."\n\
echo "Current directory: $(pwd)"\n\
echo "Checking raw data files..."\n\
ls -la data/raw/ 2>/dev/null || echo "No raw data found"\n\
\n\
# Run DVC setup if needed\n\
if [ ! -f ".dvc/config" ]; then\n\
    echo "Running DVC setup..."\n\
    /app/setup_dvc.sh\n\
fi\n\
\n\
# Handle different commands\n\
case "$1" in\n\
    repro)\n\
        echo "Running: dvc repro --force"\n\
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
    setup)\n\
        echo "Running DVC setup..."\n\
        /app/setup_dvc.sh\n\
        ;;\n\
    clean)\n\
        echo "Cleaning DVC cache and outputs..."\n\
        dvc gc --workspace --force\n\
        dvc repro --force\n\
        ;;\n\
    *)\n\
        echo "Running default: dvc repro --force"\n\
        dvc repro --force\n\
        ;;\n\
esac\n\
\n\
# Display results\n\
echo "Pipeline execution completed!"\n\
echo "Output files:"\n\
find data/processed data/features models -type f 2>/dev/null | head -10 || echo "No output files found"\n\
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
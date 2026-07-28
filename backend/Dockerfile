# ML inference backend: FastAPI + uvicorn.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies in a separate layer for caching.
# Keep this list in sync with pyproject.toml [project.dependencies].
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7" \
        "pydantic-settings>=2.3" "joblib>=1.4" "scikit-learn~=1.9.0" \
        "pandas>=2.2" "numpy>=2.0" "python-multipart>=0.0.9" "openpyxl>=3.1"

COPY app ./app
COPY models ./models

# Drop privileges: run the service as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

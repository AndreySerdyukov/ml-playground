# Backend ML-инференса: FastAPI + uvicorn.
FROM python:3.12-slim

WORKDIR /app

# Устанавливаем зависимости отдельным слоем для кэша.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        "fastapi>=0.111" "uvicorn[standard]>=0.30" "pydantic>=2.7" \
        "pydantic-settings>=2.3" "joblib>=1.4" "scikit-learn>=1.5" \
        "pandas>=2.2" "numpy>=1.26"

COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Image for the API and the hourly forecaster (same image, different command; see docker-compose.yml).
# Contains the code and the libraries to RUN a model. It does not contain data, models, secrets or MLflow.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Exact library versions the model was trained with (made by docker/make_constraints.py).
# A model saved by one LightGBM / scikit-learn version should be loaded by the same version.
COPY docker/constraints.txt /tmp/constraints.txt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e .

# models/ and data/forecasts/ are mounted by docker-compose.yml. The key comes from the environment.
CMD ["uvicorn", "hawa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

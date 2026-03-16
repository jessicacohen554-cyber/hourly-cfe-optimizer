FROM python:3.11-slim

WORKDIR /app

# Install system deps for numba
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic profiles if real data not present
RUN python scripts/generate_synthetic_profiles.py 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

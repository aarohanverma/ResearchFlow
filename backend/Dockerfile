FROM python:3.11-slim

WORKDIR /app

# System deps needed by marker-pdf and numpy
RUN apt-get update && apt-get install -y \
    build-essential libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

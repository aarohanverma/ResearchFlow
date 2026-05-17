# ResearchFlow — Backend Development Image
#
# Used by the root-level docker-compose.yml for hot-reload local development.
# The build/Dockerfile.backend image is the slimmer, multi-stage production
# variant used by build/docker-compose.yml.
#
# Both images carry the same runtime system deps so dev and prod parse PDFs
# and generate media the same way:
#   libgomp1                     — OpenMP runtime for hdbscan / numpy parallelism
#   libgl1 libglib2.0-0 libsm6 libxext6 — OpenCV headless (marker-pdf)
#   build-essential libpq-dev    — compile asyncpg / hdbscan native extensions
#   ffmpeg                       — audio segment concatenation for podcast generation
#   nodejs + @marp-team/marp-cli — slide rendering for the slides workflow
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
        curl \
        ca-certificates \
        gnupg \
        git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @marp-team/marp-cli@latest \
    && apt-get purge -y curl gnupg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

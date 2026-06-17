# Single image used by all three services (api, streamlit, scheduler).
# CMD is overridden per-service in docker-compose.yml.

FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by PyTorch (OpenMP threading)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first — avoids pulling the 6 GB CUDA build
# that `pip install torch` would download by default.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure runtime dirs exist for bare `docker run` (bind mounts shadow these in compose)
RUN mkdir -p data models output config

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["docker-entrypoint.sh"]

# Default command — overridden in docker-compose.yml for streamlit and scheduler
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

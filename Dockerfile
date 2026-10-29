# Container for the FastAPI backend (the Streamlit UI can use the same image).
#
# Build:  docker build -t rag-assistant .
# Run:    docker run --env-file .env -p 8000:8000 rag-assistant
# UI:     docker run --env-file .env -p 8501:8501 -e API_URL=http://host.docker.internal:8000 \
#             rag-assistant streamlit run app/streamlit_app.py --server.address=0.0.0.0
#
# The vector store is built on first start if it is missing (needs internet to
# download the Wikipedia articles). Mount a volume on /app/chroma_db to keep it.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml config.yaml ./
COPY src ./src
COPY scripts ./scripts
COPY app ./app
COPY data/poisoned ./data/poisoned
RUN pip install --no-cache-dir -e .

# Download the embedding model at build time so the container starts fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["sh", "scripts/docker_start.sh"]

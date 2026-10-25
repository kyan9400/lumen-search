# syntax=docker/dockerfile:1.7
# Lumen Search: API + web UI, with Tesseract (English and Russian) for scanned pages.

FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only PyTorch: the CUDA build from PyPI adds several gigabytes the API never uses
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Bake the models into the image so the container starts without network access.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')" \
 && if [ -n "${RERANKER_MODEL}" ]; then \
      python -c "from sentence_transformers import CrossEncoder as C; C('${RERANKER_MODEL}')"; \
    fi


FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /app lumen
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=lumen /opt/hf /opt/hf
WORKDIR /app
COPY --chown=lumen corpus ./corpus
COPY --chown=lumen eval ./eval
RUN mkdir /data && chown lumen /data
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    LUMEN_DATA_DIR=/data
USER lumen
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["lumen", "serve", "--host", "0.0.0.0", "--port", "8000"]

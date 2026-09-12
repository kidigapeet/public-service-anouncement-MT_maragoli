# CPU image. Small, runs anywhere, several seconds per sentence per system.
# For a GPU box use Dockerfile.gpu instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# torch first, from the CPU index, so pip does not pull ~2.5 GB of CUDA wheels
# onto a machine that has no GPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static ./static
COPY metrics ./metrics

# The weights are ~7 GB across four models. Mount /cache as a volume or every
# container restart re-downloads them.
RUN mkdir -p /cache/huggingface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/home/appuser/.cache/huggingface
COPY app.py rag.py rag_metrics.py ragas_eval.py continuous_eval.py guardrails.py tracing.py provider.py telemetry.py ./
COPY templates ./templates
COPY static ./static
COPY evals/results ./evals/results
COPY evals/cases.json ./evals/cases.json
RUN mkdir -p /app/instance && chown appuser:appuser /app/instance
USER appuser
RUN python -c "from rag import encoder; encoder(); from app import app"
EXPOSE 7860
CMD ["gunicorn","--bind","0.0.0.0:7860","--workers","1","--threads","4","--timeout","240","app:app"]

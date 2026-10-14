FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader -d "$NLTK_DATA" punkt stopwords

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/instance/uploads \
    && chown -R appuser:appuser /app/instance

COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "wsgi:application"]

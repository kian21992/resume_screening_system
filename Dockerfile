FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/kian21992/resume_screening_system"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && python -m nltk.downloader -d /opt/nltk_data punkt stopwords \
    && python -c "import spacy; from nltk.corpus import stopwords; from nltk.tokenize import word_tokenize; spacy.load('en_core_web_sm'); assert stopwords.words('english'); assert word_tokenize('Docker build check.')"

COPY . .
RUN mkdir -p /app/instance/uploads \
    && chown -R appuser:appuser /app/instance \
    && chmod +x /app/scripts/docker-entrypoint.sh

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/login', timeout=4)" || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "wsgi:application"]

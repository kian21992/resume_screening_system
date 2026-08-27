#!/usr/bin/env bash
set -euo pipefail

python -m flask --app app migrate-device-isolation
python -m flask --app app bootstrap-admin

exec gunicorn \
    --bind "0.0.0.0:${PORT:-10000}" \
    --workers 1 \
    --threads 2 \
    --timeout 180 \
    --access-logfile - \
    --error-logfile - \
    wsgi:application

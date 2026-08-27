#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader -d /opt/render/project/src/nltk_data punkt stopwords

#!/bin/sh
set -eu

python -m flask --app app migrate-device-isolation

exec "$@"

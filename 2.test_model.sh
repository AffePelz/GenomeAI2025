#!/usr/bin/env bash
set -eu

echo "Activating virtual environment"
. ".venv/bin/activate"

python "main/GPN_demo.py"
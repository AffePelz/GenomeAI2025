#!/usr/bin/env bash
set -eu

echo "Activating virtual environment"
. ".venv/bin/activate"

python main/kmer_dataset.py
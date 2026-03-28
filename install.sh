#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

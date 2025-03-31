#!/usr/bin/env bash

set -e

eval "$(conda shell.bash hook)"

if conda info --envs | grep -q '^physgame-benchmark '; then
    conda activate physgame-benchmark
else
    conda create -n physgame-benchmark -y python=3.12
    conda activate physgame-benchmark
    pip install -r requirements.txt
    pip install -e .
fi

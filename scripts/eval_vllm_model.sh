#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

python eval_vllm_model.py $@

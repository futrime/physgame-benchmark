#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

python eval_local_model.py $@

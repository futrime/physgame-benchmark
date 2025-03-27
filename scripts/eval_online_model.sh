#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

python eval_online_model.py $@

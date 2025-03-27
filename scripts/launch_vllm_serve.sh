#!/usr/bin/env bash

set -e

source scripts/prepare_env.sh

vllm serve $@

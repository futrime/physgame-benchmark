# physgame-benchmark

A re-implmentation of [PhysGame Benchmark](https://github.com/PhysGame/PhysGame) for evaluating the performance of online and local VLMs.

## Install

```bash
source scripts/prepare_env.sh
```

## Usage

Evaluate GPT-4o online model with 0-shot profile and batch size 8:

```bash
bash scripts/eval_online_model.sh \
    --profile zero_shot \
    --model gpt-4o \
    --model-provider openai \
    --batch-size 8
```

Evaluate Qwen2.5-VL-7B-Instruct local model with 0-shot profile and batch size 8:

```bash
bash scripts/eval_vllm_model.sh \
    --profile zero_shot \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --batch-size 8
```

## Contributing

PRs accepted.

## License

MIT © Zijian Zhang

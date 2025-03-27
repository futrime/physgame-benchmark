import argparse
import asyncio
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, cast

import dotenv
import langchain
import langchain.chat_models
import langchain.utils
import tqdm
import tqdm.asyncio
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from torch.utils.data import DataLoader, Subset

from benchmark import Dataset, profiles

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_DIR = ".dev/eval"


@dataclass
class EvalConfig:
    profile: str

    model: str
    model_provider: str
    base_url: Optional[str]

    batch_size: int
    dataset_dir: pathlib.Path
    eval_result_dir: pathlib.Path
    rate_limit_rpm: Optional[int]
    video_sample_num_frames: int

    @property
    def name(self) -> str:
        return f"{self.model.replace('/', '-')}-{self.profile}"


class EvalResult(TypedDict):
    accuracy: float
    invalid_ratio: float
    accuracy_by_classes: Dict[str, float]
    invalid_ratio_by_classes: Dict[str, float]


class ModelOutputEntry(TypedDict):
    question_id: str
    prediction: str


async def evaluate(eval_config: EvalConfig) -> None:
    os.makedirs(
        os.path.join(eval_config.eval_result_dir, eval_config.name), exist_ok=True
    )

    # If the metrics file already exists, skip evaluation.
    if os.path.exists(
        os.path.join(eval_config.eval_result_dir, eval_config.name, "metrics.json")
    ):
        return

    dataset = Dataset(eval_config.dataset_dir)

    # Load existing model output IDs.
    generated_question_ids: List[str] = []
    if os.path.exists(
        os.path.join(eval_config.eval_result_dir, eval_config.name, "output.jsonl")
    ):
        with open(
            os.path.join(eval_config.eval_result_dir, eval_config.name, "output.jsonl"),
            encoding="utf-8",
        ) as f:
            for line in f:
                model_output_entry = json.loads(line)
                generated_question_ids.append(model_output_entry["question_id"])

    dataloader = DataLoader(
        Subset(
            dataset,
            [
                i
                for i in range(len(dataset))
                if dataset[i].question_id not in generated_question_ids
            ],
        ),
        batch_size=eval_config.batch_size,
        collate_fn=lambda x: x,
    )

    profile = profiles.get_profile(eval_config.profile)

    chat_model = langchain.chat_models.init_chat_model(
        eval_config.model,
        model_provider=eval_config.model_provider,
        base_url=eval_config.base_url,
        max_tokens=1024,
        rate_limiter=(
            None
            if eval_config.rate_limit_rpm is None
            else InMemoryRateLimiter(
                requests_per_second=eval_config.rate_limit_rpm / 60
            )
        ),
        temperature=0,
    )

    # Stage 1: Generate outputs.
    for dataset_entries in tqdm.tqdm(dataloader):
        dataset_entries: List[Dataset.Entry]

        existing_messages: List[List[BaseMessage]] = [
            [] for _ in range(len(dataset_entries))
        ]

        while True:
            messages = [
                profile.build_prompt(
                    dataset_entry,
                    existing_messages=existing_messages[i],
                )
                for i, dataset_entry in enumerate(dataset_entries)
            ]

            if not any(message is not None for message in messages):
                break

            # Only call the model for those not None.
            id_map: List[int] = [
                i for i, message in enumerate(messages) if message is not None
            ]

            responses = await chat_model.abatch(
                cast(
                    List[Any], [message for message in messages if message is not None]
                )
            )

            for i, response in zip(id_map, responses):
                existing_messages[i].append(response)

        for dataset_entry, messages in zip(dataset_entries, existing_messages):
            assert len(messages) > 0

            response_content = str(messages[-1].content)

            with open(
                os.path.join(
                    eval_config.eval_result_dir, eval_config.name, "output.jsonl"
                ),
                "a",
                encoding="utf-8",
            ) as f:
                json.dump(
                    ModelOutputEntry(
                        question_id=dataset_entry.question_id,
                        prediction=response_content,
                    ),
                    f,
                )
                f.write("\n")

    # Stage 2: Evaluate outputs.
    count: int = 0
    valid_count: int = 0
    correct_count: int = 0

    count_by_tags: Dict[str, int] = {}
    valid_count_by_tags: Dict[str, int] = {}
    correct_count_by_tags: Dict[str, int] = {}

    # Build an index for fast lookup of dataset entries by question ID.
    dataset_index = {entry.question_id: entry for entry in dataset}
    with open(
        os.path.join(eval_config.eval_result_dir, eval_config.name, "output.jsonl"),
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            model_output_entry: ModelOutputEntry = json.loads(line)
            dataset_entry = dataset_index.get(model_output_entry["question_id"])
            if dataset_entry is None:
                continue

            count += 1

            for tag in dataset_entry.tags:
                count_by_tags.setdefault(tag, 0)
                count_by_tags[tag] += 1

            correctness = profile.check_response(
                dataset_entry, model_output_entry["prediction"]
            )

            if correctness is None:
                continue

            valid_count += 1
            correct_count += 1 if correctness else 0

            for tag in dataset_entry.tags:
                valid_count_by_tags.setdefault(tag, 0)
                correct_count_by_tags.setdefault(tag, 0)

                valid_count_by_tags[tag] += 1
                correct_count_by_tags[tag] += 1 if correctness else 0

    eval_result = EvalResult(
        accuracy=correct_count / len(dataset),
        invalid_ratio=1 - valid_count / len(dataset),
        accuracy_by_classes={
            tag: correct_count_by_tags[tag] / count_by_tags[tag]
            for tag in valid_count_by_tags
        },
        invalid_ratio_by_classes={
            tag: 1 - valid_count_by_tags[tag] / count_by_tags[tag]
            for tag in valid_count_by_tags
        },
    )

    with open(
        os.path.join(eval_config.eval_result_dir, eval_config.name, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(eval_result, f, indent=4)


async def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        choices=profiles.AVAILABLE_PROFILES,
        help="Evaluation profile",
    )

    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--model-provider", required=True, help="Model provider")
    parser.add_argument("--base-url", default=None, help="Base URL")

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory"
    )
    parser.add_argument(
        "--eval-result-dir",
        default=DEFAULT_EVAL_RESULT_DIR,
        help="Evaluation result directory",
    )
    parser.add_argument(
        "--rate-limit-rpm", type=int, default=0, help="Rate limit (RPM)"
    )
    parser.add_argument(
        "--video-sample-num-frames",
        type=int,
        default=8,
        help="Number of frames to sample for a video",
    )

    args = parser.parse_args()

    eval_config = EvalConfig(
        profile=cast(str, args.profile),
        model=cast(str, args.model),
        model_provider=cast(str, args.model_provider),
        base_url=cast(Optional[str], args.base_url),
        batch_size=cast(int, args.batch_size),
        dataset_dir=pathlib.Path(cast(str, args.dataset_dir)),
        eval_result_dir=pathlib.Path(cast(str, args.eval_result_dir)),
        rate_limit_rpm=cast(int, args.rate_limit_rpm) or None,
        video_sample_num_frames=cast(int, args.video_sample_num_frames),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

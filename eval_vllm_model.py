import argparse
import asyncio
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, TypedDict, cast

import dotenv
import langchain_core.messages
import tqdm
import tqdm.asyncio
from langchain_core.messages import AIMessage, BaseMessage
from torch.utils.data import DataLoader, Subset
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from benchmark import Dataset, profiles

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_DIR = ".dev/eval"


@dataclass
class EvalConfig:
    profile: str

    model: str

    batch_size: int
    dataset_dir: pathlib.Path
    eval_result_dir: pathlib.Path
    pipeline_parallel_size: int
    tensor_parallel_size: int

    @property
    def name(self) -> str:
        return f"{self.model.split('/')[-1]}-{self.profile}"


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

    llm = LLM(
        eval_config.model,
        limit_mm_per_prompt={"image": profile.video_sample_num_frames},
        pipeline_parallel_size=eval_config.pipeline_parallel_size,
        tensor_parallel_size=eval_config.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0)

    # Stage 1: Generate outputs.
    for dataset_entries in tqdm.tqdm(dataloader):
        dataset_entries: List[Dataset.Entry]

        existing_messages: List[List[BaseMessage]] = [
            [] for _ in range(len(dataset_entries))
        ]

        while True:
            messages = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        profile.build_prompt,
                        dataset_entry,
                        existing_messages=existing_messages[i],
                    )
                    for i, dataset_entry in enumerate(dataset_entries)
                ]
            )

            for i, messages_entry in enumerate(messages):
                if messages_entry is None:
                    continue

                existing_messages[i] = messages_entry

            if not any(messages_entry is not None for messages_entry in messages):
                break

            # Only call the model for those not None.
            id_map: List[int] = [
                i
                for i, messages_entry in enumerate(messages)
                if messages_entry is not None
            ]

            messages_for_generation = cast(
                List[List[BaseMessage]],
                [message for message in messages if message is not None],
            )

            openai_messages = cast(
                List[List[ChatCompletionMessageParam]],
                [
                    langchain_core.messages.convert_to_openai_messages(message)
                    for message in messages_for_generation
                ],
            )

            outputs = llm.chat(
                openai_messages,
                sampling_params=sampling_params,
            )

            for i, output in zip(id_map, outputs):
                generated_text = output.outputs[0].text
                response_message = AIMessage(generated_text)
                existing_messages[i].append(response_message)

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
        choices=profiles.get_available_profiles(),
        help="Evaluation profile",
    )

    parser.add_argument("--model", required=True, help="Model to use")

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
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of pipeline parallel groups",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of tensor parallel groups",
    )

    args = parser.parse_args()

    eval_config = EvalConfig(
        profile=cast(str, args.profile),
        model=cast(str, args.model),
        batch_size=cast(int, args.batch_size),
        dataset_dir=pathlib.Path(cast(str, args.dataset_dir)),
        eval_result_dir=pathlib.Path(cast(str, args.eval_result_dir)),
        pipeline_parallel_size=cast(int, args.pipeline_parallel_size),
        tensor_parallel_size=cast(int, args.tensor_parallel_size),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

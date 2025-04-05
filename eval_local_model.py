import argparse
import asyncio
import pathlib
from dataclasses import dataclass
from typing import Dict, List, cast

import accelerate
import dotenv
import torch
import tqdm
from openai.types.chat import ChatCompletionMessageParam
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoProcessor, GenerationMixin, ProcessorMixin

from physgame_benchmark import Dataset, DatasetEntry, profiles
from physgame_benchmark.result_manager import ModelOutputEntry, ResultManager

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_BASE_DIR = ".dev/eval"


@dataclass
class EvalConfig:
    profile: str

    model: str

    batch_size: int
    dataset_dir: pathlib.Path
    result_base_dir: pathlib.Path

    @property
    def name(self) -> str:
        return f"{self.model.replace('/', '-')}-{self.profile}"


async def evaluate(eval_config: EvalConfig) -> None:
    profile = profiles.get_profile(eval_config.profile)

    result_manager = ResultManager(eval_config.result_base_dir / eval_config.name)
    result_manager.load_model_outputs()

    processor: ProcessorMixin = AutoProcessor.from_pretrained(eval_config.model)

    with accelerate.init_empty_weights():
        model: GenerationMixin = AutoModel.from_pretrained(
            eval_config.model,
            torch_dtype="auto",
        )

    @torch.inference_mode()
    async def generate(inputs: List[List[ChatCompletionMessageParam]]) -> List[str]:
        processed_inputs = processor.apply_chat_template(
            cast(List[List[Dict[str, str]]], inputs),
            add_generation_prompt=True,
        )

        outputs = model.generate(
            **processed_inputs,
            do_sample=False,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=1,
        )

    dataset = Dataset(eval_config.dataset_dir)
    dataloader = DataLoader(
        dataset=Subset(
            dataset,
            [
                i
                for i in range(len(dataset))
                if dataset[i].question_id not in result_manager.model_outputs.keys()
            ],
        ),
        batch_size=eval_config.batch_size,
        collate_fn=lambda x: x,
    )

    for dataset_entries in tqdm.tqdm(dataloader):
        dataset_entries: List[DatasetEntry]

        predictions = await profile.predict(dataset_entries, generate)
        assert len(predictions) == len(dataset_entries)

        for dataset_entry, prediction in zip(dataset_entries, predictions):
            question_id = dataset_entry.question_id
            result_manager.add_model_output(
                ModelOutputEntry(
                    question_id=question_id,
                    prediction=prediction,
                )
            )

        result_manager.save_model_outputs()

    result_manager.generate_metrics(dataset, profile.check_answer)


async def main() -> None:
    dotenv.load_dotenv(override=True)

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
        "--result-base-dir",
        default=DEFAULT_EVAL_RESULT_BASE_DIR,
        help="Evaluation result base directory",
    )

    args = parser.parse_args()

    eval_config = EvalConfig(
        profile=cast(str, args.profile),
        model=cast(str, args.model),
        batch_size=cast(int, args.batch_size),
        dataset_dir=pathlib.Path(cast(str, args.dataset_dir)),
        result_base_dir=pathlib.Path(cast(str, args.result_base_dir)),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

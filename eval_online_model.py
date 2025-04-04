import argparse
import asyncio
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, cast

import dotenv
import openai
import tqdm
from openai.types.responses import ResponseInputParam
from torch.utils.data import DataLoader, Subset

from physgame_benchmark import Dataset, DatasetEntry, profiles
from physgame_benchmark.result_manager import ResultManager

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_BASE_DIR = ".dev/eval"


@dataclass
class EvalConfig:
    profile: str

    model: str
    base_url: Optional[str]

    batch_size: int
    dataset_dir: pathlib.Path
    result_base_dir: pathlib.Path

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
    profile = profiles.utils.get_profile(eval_config.profile)

    result_manager = ResultManager(eval_config.result_base_dir / eval_config.name)
    result_manager.load_model_outputs()

    client = openai.AsyncOpenAI(
        base_url=eval_config.base_url,
        max_retries=10,
    )

    async def generate(inputs: List[ResponseInputParam]) -> List[str]:
        tasks = [
            client.responses.create(
                input=input,
                model=eval_config.model,
                temperature=0,
            )
            for input in inputs
        ]
        responses = await asyncio.gather(*tasks)
        return [response.output_text for response in responses]

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
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        choices=profiles.utils.get_available_profiles(),
        help="Evaluation profile",
    )

    parser.add_argument("--model", required=True, help="Model to use")
    parser.add_argument("--base-url", default=None, help="Base URL")

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
        base_url=cast(Optional[str], args.base_url),
        batch_size=cast(int, args.batch_size),
        dataset_dir=pathlib.Path(cast(str, args.dataset_dir)),
        result_base_dir=pathlib.Path(cast(str, args.result_base_dir)),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

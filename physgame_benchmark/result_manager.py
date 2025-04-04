import json
import logging
import os
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, Optional, TypedDict

from torch.utils.data import Dataset as TorchDataset

from .dataset import DatasetEntry

_MODEL_OUTPUT_FILE_NAME = "output.jsonl"
_METRICS_FILE_NAME = "metrics.json"


class Metrics(TypedDict):
    accuracy: float
    valid_rate: float
    accuracy_by_tag: Dict[str, float]
    valid_rate_by_tag: Dict[str, float]


class ModelOutputEntry(TypedDict):
    question_id: str
    prediction: str


class ResultManager:
    _logger: Logger
    _model_outputs: Dict[str, ModelOutputEntry] = {}
    _result_dir: Path

    def __init__(self, result_dir: Path):
        """Initializes ResultManager.
        Args:
            result_dir: Directory to store the result files.
        """

        self._logger = logging.getLogger(__name__)
        self._result_dir = result_dir

    @property
    def model_outputs(self) -> Dict[str, ModelOutputEntry]:
        """Gets model outputs.

        Returns:
            A dictionary of model outputs.
        """

        return self._model_outputs

    def add_model_output(self, model_output_entry: ModelOutputEntry) -> None:
        """Adds a model output entry to the result manager.

        This will overwrite any existing model output entry with the same question_id.

        Args:
            model_output_entry: The model output entry to add.
        """

        self._model_outputs[model_output_entry["question_id"]] = model_output_entry

    def load_model_outputs(self) -> Dict[str, ModelOutputEntry]:
        """Loads model outputs from the result directory.

        This will overwrite any existing model outputs in memory.

        Returns:
            A dictionary of model outputs.
        """

        model_output_file = self._result_dir / _MODEL_OUTPUT_FILE_NAME

        if model_output_file.exists():
            with open(model_output_file, "r", encoding="utf-8") as f:
                for line in f:
                    model_output_entry: ModelOutputEntry = json.loads(line)
                    self._model_outputs[model_output_entry["question_id"]] = (
                        model_output_entry
                    )

        return self._model_outputs

    def save_model_outputs(self) -> None:
        """Saves model outputs to the result directory.

        This will overwrite any existing model outputs in the file.
        """

        model_output_file = self._result_dir / _MODEL_OUTPUT_FILE_NAME

        os.makedirs(self._result_dir, exist_ok=True)

        with open(model_output_file, "w", encoding="utf-8") as f:
            for model_output_entry in self._model_outputs.values():
                json.dump(model_output_entry, f)
                f.write("\n")

    def generate_metrics(
        self,
        dataset: TorchDataset[DatasetEntry],
        check_answer_func: Callable[[str, DatasetEntry], Optional[bool]],
    ) -> Metrics:
        """Generates and saves metrics based on the model outputs.

        This will overwrite any existing metrics in the file.

        Args:
            dataset: The dataset to use for generating metrics.
            check_answer_func: A function that checks if the answer is correct.

        Returns:
            The generated metrics.
        """

        dataset_entries = {entry.question_id: entry for entry in dataset}

        # Calculate metrics.

        count = 0
        valid_count = 0
        correct_count = 0

        count_by_tag: Dict[str, int] = {}
        valid_count_by_tag: Dict[str, int] = {}
        correct_count_by_tag: Dict[str, int] = {}

        for model_output in self._model_outputs.values():
            question_id = model_output["question_id"]

            if question_id not in dataset_entries:
                self._logger.warning(
                    f"Model output with question_id {question_id} not in dataset. Skipping..."
                )
                continue

            dataset_entry = dataset_entries[question_id]

            count += 1
            for tag in dataset_entry.tags:
                count_by_tag.setdefault(tag, 0)
                count_by_tag[tag] += 1

            correct = check_answer_func(model_output["prediction"], dataset_entry)
            if correct is None:  # Invalid answer.
                continue

            valid_count += 1
            correct_count += 1 if correct else 0

            for tag in dataset_entry.tags:
                valid_count_by_tag.setdefault(tag, 0)
                valid_count_by_tag[tag] += 1

                correct_count_by_tag.setdefault(tag, 0)
                correct_count_by_tag[tag] += 1 if correct else 0

        metrics = Metrics(
            accuracy=correct_count / count if count > 0 else 0,
            valid_rate=valid_count / count if count > 0 else 0,
            accuracy_by_tag={
                tag: correct_count_by_tag[tag] / count_by_tag[tag]
                for tag in count_by_tag
            },
            valid_rate_by_tag={
                tag: valid_count_by_tag[tag] / count_by_tag[tag] for tag in count_by_tag
            },
        )

        # Save metrics to file.

        metrics_file = self._result_dir / _METRICS_FILE_NAME

        os.makedirs(self._result_dir, exist_ok=True)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        return metrics

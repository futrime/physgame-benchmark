from abc import ABC, abstractmethod
from typing import Awaitable, Callable, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from ..dataset import DatasetEntry


class BaseProfile(ABC):
    @property
    @abstractmethod
    def num_video_sample_frames(self) -> int:
        """The number of frames to sample from the video."""

        raise NotImplementedError

    @abstractmethod
    async def predict(
        self,
        dataset_entries: List[DatasetEntry],
        generate_func: Callable[
            [List[List[ChatCompletionMessageParam]]], Awaitable[List[str]]
        ],
    ) -> List[str]:
        """Generates predictions for the given dataset entries.

        Args:
            dataset_entries: The dataset entries to build the prompt for.
            generate_func: The function to call to generate the response from the model.

        Returns:
            The model final predictions.
        """

        raise NotImplementedError

    @abstractmethod
    def check_answer(
        self, predicted: str, dataset_entry: DatasetEntry
    ) -> Optional[bool]:
        """Checks the model response to the given dataset entry.

        Args:
            predicted: The model response to check.
            dataset_entry: The dataset entry for which the model response should be checked.

        Returns:
            True if the response is correct, False if the response is incorrect, or None if the response is invalid.
        """

        raise NotImplementedError

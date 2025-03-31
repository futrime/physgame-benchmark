from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..dataset import DatasetEntry


class BaseProfile(ABC):
    @property
    @abstractmethod
    def video_sample_num_frames(self) -> int:
        """The number of frames to sample from the video."""

        raise NotImplementedError

    @abstractmethod
    def build_prompt(
        self, dataset_entry: DatasetEntry, existing_messages: List[BaseMessage]
    ) -> Optional[List[BaseMessage]]:
        """Builds the prompt for the given dataset entry and existing messages.

        Args:
            dataset_entry: The dataset entry for which to build the prompt.
            existing_messages: The messages already present in the conversation.

        Returns:
            A list of messages to send to the user, or None if no messages should be sent.
        """

        raise NotImplementedError

    @abstractmethod
    def check_response(
        self, dataset_entry: DatasetEntry, response: str
    ) -> Optional[bool]:
        """Checks the model response to the given dataset entry.

        Args:
            dataset_entry: The dataset entry for which the model response should be checked.
            response: The model response.

        Returns:
            True if the response is correct, False if the response is incorrect, or None if the response is invalid.
        """

        raise NotImplementedError

    @staticmethod
    def _get_round(messages: List[BaseMessage]) -> int:
        """Gets the round number from the given messages.

        Args:
            messages: The messages to get the round number from.

        Returns:
            The round number.
        """

        if len(messages) == 0:
            return 0

        assert len(messages) % 2 == 1
        assert isinstance(messages[0], SystemMessage)

        round = 0

        for i in range(1, len(messages), 2):
            assert isinstance(messages[i], HumanMessage)
            assert isinstance(messages[i + 1], AIMessage)

            round += 1

        return round

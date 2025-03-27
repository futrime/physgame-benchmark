from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import BaseMessage

from benchmark.dataset import Dataset


class BaseProfile(ABC):
    @abstractmethod
    def build_prompt(
        self, dataset_entry: Dataset.Entry, existing_messages: List[BaseMessage]
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
        self, dataset_entry: Dataset.Entry, response: str
    ) -> Optional[bool]:
        """Checks the model response to the given dataset entry.

        Args:
            dataset_entry: The dataset entry for which the model response should be checked.
            response: The model response.

        Returns:
            True if the response is correct, False if the response is incorrect, or None if the response is invalid.
        """

        raise NotImplementedError

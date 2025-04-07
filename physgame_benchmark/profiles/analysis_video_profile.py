import asyncio
import re
from typing import Awaitable, Callable, List, Optional

from ..conversation import Conversation, Message, TextContentPart, VideoContentPart
from ..dataset import DatasetEntry
from .base_profile import BaseProfile

_NUM_FRAMES = 8


class AnalysisVideoProfile(BaseProfile):
    @property
    def num_frames(self) -> int:
        return _NUM_FRAMES

    async def predict(
        self,
        dataset_entries: List[DatasetEntry],
        generate_func: Callable[[List[Conversation]], Awaitable[List[str]]],
    ) -> List[str]:
        inputs_0 = await asyncio.gather(
            *[
                self._prepare_input_round_0(dataset_entry)
                for dataset_entry in dataset_entries
            ]
        )

        outputs_0 = await generate_func(inputs_0)

        inputs_1 = await asyncio.gather(
            *[
                self._prepare_input_round_1(dataset_entry, input_0, output_0)
                for dataset_entry, input_0, output_0 in zip(
                    dataset_entries, inputs_0, outputs_0
                )
            ]
        )

        outputs_1 = await generate_func(inputs_1)

        return outputs_1

    def check_answer(
        self, predicted: str, dataset_entry: DatasetEntry
    ) -> Optional[bool]:
        match = re.search(r"\(?([A-D])\)", predicted)
        if not match:
            return None

        return match.group(1) == dataset_entry.answer

    async def _prepare_input_round_0(
        self,
        dataset_entry: DatasetEntry,
    ) -> Conversation:
        response_input: Conversation = [
            Message(
                role="system",
                content=[
                    TextContentPart(
                        text="""Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and analyze every option to determine how accurate are they to describe the detected glitch.""",
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    VideoContentPart(
                        path=dataset_entry.video_path,
                    ),
                    TextContentPart(
                        text=f"""{dataset_entry.question}
{"\n".join([f"({key}) {value}" for key, value in dataset_entry.options.items()])}
Give analysis for every option, but do not give the answer yet.""",
                    ),
                ],
            ),
        ]

        return response_input

    async def _prepare_input_round_1(
        self,
        dataset_entry: DatasetEntry,
        input_0: Conversation,
        output_0: str,
    ) -> Conversation:
        assert len(input_0) == 2

        response_input: Conversation = [
            Message(
                role="system",
                content=[
                    TextContentPart(
                        text="""Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and select the most accurate option to describe the detected glitch."""
                    )
                ],
            ),
            input_0[1],
            Message(
                role="assistant",
                content=[
                    TextContentPart(
                        text=output_0,
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    TextContentPart(
                        text=f"""{dataset_entry.question}
{"\n".join([f"({key}) {value}" for key, value in dataset_entry.options.items()])}
Only give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). \
You must always give an option, even if you are not sure.""",
                    ),
                ],
            ),
        ]

        return response_input

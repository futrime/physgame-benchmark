import asyncio
import re
from typing import Awaitable, Callable, List, Optional

from ..conversation import Conversation, Message, TextContentPart, VideoContentPart
from ..dataset import DatasetEntry
from .base_profile import BaseProfile

_NUM_VIDEO_SAMPLE_FRAMES = 8


class ZeroShotProfile(BaseProfile):
    @property
    def num_video_sample_frames(self) -> int:
        return _NUM_VIDEO_SAMPLE_FRAMES

    async def predict(
        self,
        dataset_entries: List[DatasetEntry],
        generate_func: Callable[[List[Conversation]], Awaitable[List[str]]],
    ) -> List[str]:
        inputs = await asyncio.gather(
            *[self._prepare_input(dataset_entry) for dataset_entry in dataset_entries]
        )

        return await generate_func(inputs)

    def check_answer(
        self, predicted: str, dataset_entry: DatasetEntry
    ) -> Optional[bool]:
        match = re.search(r"\(?([A-D])\)", predicted)
        if not match:
            return None

        return match.group(1) == dataset_entry.answer

    async def _prepare_input(
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
and select the most accurate option to describe the detected glitch.""",
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    VideoContentPart(
                        file_path=dataset_entry.video_path,
                        num_sample_frames=self.num_video_sample_frames,
                    ),
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

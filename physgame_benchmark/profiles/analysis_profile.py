import asyncio
import re
from typing import Awaitable, Callable, List, Optional

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from ..dataset import DatasetEntry
from ..utils import sample_video_to_base64_images
from .base_profile import BaseProfile

_NUM_VIDEO_SAMPLE_FRAMES = 8


class AnalysisProfile(BaseProfile):
    @property
    def num_video_sample_frames(self) -> int:
        return _NUM_VIDEO_SAMPLE_FRAMES

    async def predict(
        self,
        dataset_entries: List[DatasetEntry],
        generate_func: Callable[
            [List[List[ChatCompletionMessageParam]]], Awaitable[List[str]]
        ],
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
    ) -> List[ChatCompletionMessageParam]:
        base64_images = await asyncio.to_thread(
            sample_video_to_base64_images,
            dataset_entry.video_path,
            num_frames=_NUM_VIDEO_SAMPLE_FRAMES,
        )

        response_input: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="""Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and analyze every option to determine how accurate are they to describe the detected glitch.""",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    *[
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=ImageURL(
                                url=f"data:image/jpeg;base64,{base64_image}"
                            ),
                        )
                        for base64_image in base64_images
                    ],
                    ChatCompletionContentPartTextParam(
                        type="text",
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
        input_0: List[ChatCompletionMessageParam],
        output_0: str,
    ) -> List[ChatCompletionMessageParam]:
        assert len(input_0) == 2

        response_input: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="""Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and select the most accurate option to describe the detected glitch.""",
            ),
            input_0[1],
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=output_0,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=f"""{dataset_entry.question}
{"\n".join([f"({key}) {value}" for key, value in dataset_entry.options.items()])}
Only give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). \
You must always give an option, even if you are not sure.""",
                    ),
                ],
            ),
        ]

        return response_input

import asyncio
import base64
import io
import re
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextParam,
)

from ..dataset import DatasetEntry
from ..utils import sample_video
from .base_profile import BaseProfile

_NUM_VIDEO_SAMPLE_FRAMES = 8


class ZeroShotProfile(BaseProfile):
    @property
    def num_video_sample_frames(self) -> int:
        return _NUM_VIDEO_SAMPLE_FRAMES

    async def predict(
        self,
        dataset_entries: List[DatasetEntry],
        generate_func: Callable[[List[ResponseInputParam]], Awaitable[List[str]]],
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
    ) -> ResponseInputParam:
        base64_images = await asyncio.to_thread(
            self.video_to_base64_images, dataset_entry.video_path
        )

        response_input: ResponseInputParam = [
            EasyInputMessageParam(
                role="system",
                content="""Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and select the most accurate option to describe the detected glitch.""",
            ),
            EasyInputMessageParam(
                role="user",
                content=[
                    *[
                        ResponseInputImageParam(
                            detail="auto",
                            type="input_image",
                            image_url=f"data:image/jpeg;base64,{base64_image}",
                        )
                        for base64_image in base64_images
                    ],
                    ResponseInputTextParam(
                        type="input_text",
                        text=f"""{dataset_entry.question}
{"\n".join([f"({key}) {value}" for key, value in dataset_entry.options.items()])}
Only give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D). \
You must always give an option, even if you are not sure.""",
                    ),
                ],
            ),
        ]

        return response_input

    def video_to_base64_images(self, video_path: Path) -> List[str]:
        images = sample_video(video_path, num_frames=_NUM_VIDEO_SAMPLE_FRAMES)

        base64_images: List[str] = []
        for image in images:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            base64_images.append(
                base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            )

        return base64_images

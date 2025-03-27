import base64
import io
import re
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from benchmark.dataset import Dataset
from benchmark.profiles.base_profile import BaseProfile
from benchmark.utils import sample_video

_SYSTEM_PROMPT = """Watch the video carefully and analyze the events and object movements, \
focusing on any inconsistencies with physical laws. \
Identify and highlight instances where the behavior deviates from expected real-world physics, \
and select the most accurate option to describe the detected glitch."""

_VIDEO_SAMPLE_NUM_FRAMES = 8


class ZeroShotProfile(BaseProfile):
    def build_prompt(
        self, dataset_entry: Dataset.Entry, existing_messages: List[BaseMessage]
    ) -> Optional[List[BaseMessage]]:
        if len(existing_messages) > 0:
            return None

        images = sample_video(
            dataset_entry.video_path, num_frames=_VIDEO_SAMPLE_NUM_FRAMES
        )

        base64_images: List[str] = []
        for image in images:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            base64_images.append(
                base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            )

        messages: List[BaseMessage] = [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(
                [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        }
                        for base64_image in base64_images
                    ],
                    f"""{dataset_entry.question}
{"\n".join([f"({key}) {value}" for key, value in dataset_entry.options.items()])}
Only give the best option enclosed in parentheses, i.e. (A), (B), (C), or (D).""",
                ]
            ),
        ]

        return messages

    def check_response(
        self, dataset_entry: Dataset.Entry, response: str
    ) -> Optional[bool]:
        match = re.search(r"\(?([A-D])\)", response)
        if not match:
            return None

        return match.group(1) == dataset_entry.answer

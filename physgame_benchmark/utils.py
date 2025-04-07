import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import transformers.image_transforms as image_transforms
import transformers.image_utils as image_utils
from numpy.typing import NDArray
from PIL.Image import Image


async def read_video_as_pil_images(
    video_path: Path,
    num_frames: int,
) -> List[Image]:
    video, _ = await asyncio.to_thread(
        image_utils.load_video,
        str(video_path),
        num_frames=num_frames,
    )
    video: NDArray[np.uint8]
    assert video.shape[0] == num_frames

    images: List[Image] = await asyncio.gather(
        *[
            asyncio.to_thread(
                image_transforms.to_pil_image,
                video[i],
            )
            for i in range(num_frames)
        ]
    )

    return images


async def convert_pil_image_to_base64_url(image: Image) -> str:
    def convert(image: Image) -> str:
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    return await asyncio.to_thread(convert, image)

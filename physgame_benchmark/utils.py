import base64
from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import PIL.Image
from PIL.Image import Image


def sample_video(
    video_path: Path,
    *,
    num_frames: int,
) -> List[PIL.Image.Image]:
    """Samples frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        Sampled frames as a list of PIL images.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / frame_rate

    sample_times = [duration * i / num_frames for i in range(num_frames)]

    images: List[Image] = []
    for time in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(frame)

            images.append(image)

    cap.release()
    return images


def sample_video_to_base64_images(
    video_path: Path,
    *,
    num_frames: int,
) -> List[str]:
    images = sample_video(video_path, num_frames=num_frames)

    base64_images: List[str] = []
    for image in images:
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        base64_images.append(base64.b64encode(image_bytes.getvalue()).decode("utf-8"))

    return base64_images

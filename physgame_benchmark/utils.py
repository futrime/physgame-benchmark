import pathlib
from typing import List

import cv2
import PIL.Image


def sample_video(
    video_path: pathlib.Path,
    *,
    num_frames: int,
) -> List[PIL.Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / frame_rate

    sample_times = [duration * i / num_frames for i in range(num_frames)]

    images: List[PIL.Image.Image] = []
    for time in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(frame)

            images.append(image)

    cap.release()
    return images

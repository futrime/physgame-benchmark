from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

from PIL.Image import Image


@dataclass
class BaseContentPart(ABC):
    pass


@dataclass
class TextContentPart(BaseContentPart):
    text: str


@dataclass
class ImagePillowContentPart(BaseContentPart):
    image: Image


@dataclass
class VideoContentPart(BaseContentPart):
    path: Path


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: List[BaseContentPart]


type Conversation = List[Message]

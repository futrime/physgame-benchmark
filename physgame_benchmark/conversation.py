from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal


@dataclass
class BaseContentPart(ABC):
    pass


@dataclass
class TextContentPart(BaseContentPart):
    text: str


@dataclass
class VideoContentPart(BaseContentPart):
    file_path: Path
    num_frames: int


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: List[BaseContentPart]


type Conversation = List[Message]

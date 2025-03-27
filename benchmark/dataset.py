import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict

import torch.utils.data

VIDEO_DIR_NAME = "PhysGame-Benchmark"
VIDEO_FILE_NAME_TEMPLATE = "{question_id}.mp4"


class Dataset(torch.utils.data.Dataset):
    @dataclass
    class Entry:
        question_id: str
        tags: List[str]
        video_path: pathlib.Path
        question: str
        options: Dict[Literal["A", "B", "C", "D"], str]
        answer: str

    _annotations: Dict[str, "_AnnotationEntry"]
    _base_dir: pathlib.Path

    def __init__(self, base_dir: pathlib.Path):
        self._annotations = _load_annotation(base_dir)
        self._base_dir = base_dir

    def __getitem__(self, index: int) -> Entry:
        question_id = list(self._annotations.keys())[index]
        annotation = self._annotations[question_id]

        video_path = (
            self._base_dir
            / VIDEO_DIR_NAME
            / VIDEO_FILE_NAME_TEMPLATE.format(question_id=question_id)
        )

        return Dataset.Entry(
            question_id=question_id,
            tags=[annotation["class_anno"], annotation["subclass_anno"]],
            video_path=video_path,
            question=annotation["question"],
            options=annotation["options"],
            answer=annotation["answer"],
        )

    def __len__(self) -> int:
        return len(self._annotations)


class _AnnotationEntry(TypedDict):
    question_id: str
    question: str
    options: Dict[Literal["A", "B", "C", "D"], str]
    answer: str
    class_anno: str
    subclass_anno: str


def _load_annotation(base_dir: pathlib.Path) -> Dict[str, _AnnotationEntry]:
    annotation_file = base_dir / "PhysGame_880_annotation.json"

    with annotation_file.open("r", encoding="utf-8") as f:
        data: List[_AnnotationEntry] = json.load(f)

    return {entry["question_id"]: entry for entry in data}

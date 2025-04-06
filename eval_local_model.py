import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, cast

import dotenv
import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from PIL.Image import Image
from torch.utils.data import DataLoader, Subset
from transformers import image_transforms, image_utils
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils.generic import PaddingStrategy

from physgame_benchmark import (
    Conversation,
    Dataset,
    DatasetEntry,
    ModelOutputEntry,
    ResultManager,
    TextContentPart,
    VideoContentPart,
    profiles,
)
from physgame_benchmark.profiles import BaseProfile


class _BaseModelForConditionalGeneration(PreTrainedModel, GenerationMixin):
    pass


@dataclass
class EvalConfig:
    profile: str

    model: str

    attn_implementation: str
    batch_size: int
    dataset_dir: Path
    result_base_dir: Path

    @property
    def name(self) -> str:
        return f"{self.model.split("/")[-1]}-{self.profile}"


async def convert_conversation_to_hf_format(
    conversation: Conversation,
    profile: BaseProfile,
) -> List[Dict[str, Any]]:
    hf_messages: List[Dict[str, Any]] = []

    for message in conversation:
        hf_contents: List[Dict[str, Any]] = []

        for content_part in message.content:
            if isinstance(content_part, TextContentPart):
                hf_contents.append(
                    {
                        "type": "text",
                        "text": content_part.text,
                    }
                )
            elif isinstance(content_part, VideoContentPart):
                video, _ = await asyncio.to_thread(
                    image_utils.load_video,
                    str(content_part.file_path),
                    num_frames=profile.num_frames,
                )
                video: NDArray[np.uint8]

                images: List[Image] = await asyncio.gather(
                    *[
                        asyncio.to_thread(image_transforms.to_pil_image, frame)
                        for frame in video
                    ]
                )

                hf_contents.extend(
                    [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in images
                    ]
                )

        hf_message = {
            "role": message.role,
            "content": hf_contents,
        }

        hf_messages.append(hf_message)

    return hf_messages


async def evaluate(eval_config: EvalConfig) -> None:
    profile = profiles.get_profile(eval_config.profile)

    result_manager = ResultManager(eval_config.result_base_dir / eval_config.name)
    result_manager.load_model_outputs()

    processor: ProcessorMixin = AutoProcessor.from_pretrained(
        eval_config.model,
        padding_side="left",
        trust_remote_code=True,
        use_fast=True,
    )

    model: _BaseModelForConditionalGeneration = (
        AutoModelForImageTextToText.from_pretrained(
            eval_config.model,
            attn_implementation=eval_config.attn_implementation,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    )

    @torch.inference_mode()
    async def generate(conversations: List[Conversation]) -> List[str]:
        hf_conversations = await asyncio.gather(
            *[
                convert_conversation_to_hf_format(
                    conversation,
                    profile,
                )
                for conversation in conversations
            ]
        )

        model_inputs = processor.apply_chat_template(
            hf_conversations,
            add_generation_prompt=True,
            padding=PaddingStrategy.LONGEST,
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
            video_load_backend="opencv",
        )
        assert isinstance(model_inputs, BatchFeature)

        model_inputs.to(device=model.device, dtype=model.dtype)

        generated_outputs = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        assert isinstance(generated_outputs, torch.Tensor)

        decoded_outputs: List[str] = processor.post_process_image_text_to_text(
            generated_outputs[
                :, cast(torch.LongTensor, model_inputs.input_ids).shape[1] :
            ],
        )

        return decoded_outputs

    dataset = Dataset(eval_config.dataset_dir)
    dataloader = DataLoader(
        dataset=Subset(
            dataset,
            [
                i
                for i in range(len(dataset))
                if dataset[i].question_id not in result_manager.model_outputs.keys()
            ],
        ),
        batch_size=eval_config.batch_size,
        collate_fn=lambda x: x,
    )

    for dataset_entries in tqdm.tqdm(dataloader):
        dataset_entries: List[DatasetEntry]

        predictions = await profile.predict(dataset_entries, generate)
        assert len(predictions) == len(dataset_entries)

        for dataset_entry, prediction in zip(dataset_entries, predictions):
            question_id = dataset_entry.question_id
            result_manager.add_model_output(
                ModelOutputEntry(
                    question_id=question_id,
                    prediction=prediction,
                )
            )

        result_manager.save_model_outputs()

    result_manager.generate_metrics(dataset, profile.check_answer)


async def main() -> None:
    dotenv.load_dotenv(override=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        choices=profiles.get_available_profiles(),
        help="Evaluation profile",
    )

    parser.add_argument("--model", required=True, help="Model to use")

    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "flex_attention", "sdpa", "eager"],
        help="Attention implementation to use",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dataset-dir",
        default=".dev/PhysGame/PhysGame-Benchmark",
        help="Dataset directory",
    )
    parser.add_argument(
        "--result-base-dir",
        default=".dev/eval",
        help="Evaluation result base directory",
    )

    args = parser.parse_args()

    eval_config = EvalConfig(
        profile=cast(str, args.profile),
        model=cast(str, args.model),
        attn_implementation=cast(str, args.attn_implementation),
        batch_size=cast(int, args.batch_size),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        result_base_dir=Path(cast(str, args.result_base_dir)),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

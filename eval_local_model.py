import argparse
import asyncio
import base64
import os
import pathlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, cast

import dotenv
import torch
import tqdm
from PIL.Image import Image
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    GenerationMixin,
    PreTrainedModel,
    ProcessorMixin,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
)
from transformers.utils import PaddingStrategy

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

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_BASE_DIR = ".dev/eval"


class _BaseModelForCausalLM(PreTrainedModel, GenerationMixin):
    pass


@dataclass
class EvalConfig:
    profile: str

    model: str

    batch_size: int
    dataset_dir: pathlib.Path
    result_base_dir: pathlib.Path

    @property
    def name(self) -> str:
        return f"{self.model.replace('/', '-')}-{self.profile}"


async def convert_conversation_to_hf_format(
    conversation: Conversation,
) -> List[Dict[str, Any]]:
    hf_messages: List[Dict[str, Any]] = []

    for message in conversation:
        hf_contents: List[Dict[str, str]] = []

        for content_part in message.content:
            if isinstance(content_part, TextContentPart):
                hf_contents.append(
                    {
                        "type": "text",
                        "text": content_part.text,
                    }
                )
            elif isinstance(content_part, VideoContentPart):
                # images = await asyncio.to_thread(
                #     utils.sample_video,
                #     content_part.file_path,
                #     num_frames=content_part.num_sample_frames,
                # )
                # assert len(images) == content_part.num_sample_frames

                # image_base64_urls = await asyncio.gather(
                #     *[
                #         asyncio.to_thread(convert_image_to_base64_url, image)
                #         for image in images
                #     ]
                # )
                # assert len(image_base64_urls) == content_part.num_sample_frames

                # hf_contents.extend(
                #     [
                #         {
                #             "type": "image_url",
                #             "image": image_base64_url,
                #         }
                #         for image_base64_url in image_base64_urls
                #     ]
                # )

                hf_contents.append(
                    {
                        "type": "video",
                        "path": str(content_part.file_path),
                    }
                )

        hf_message = {
            "role": message.role,
            "content": hf_contents,
        }

        hf_messages.append(hf_message)

    return hf_messages


def convert_image_to_base64_url(
    image: Image,
) -> str:
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def patch_hf() -> None:
    AutoModelForCausalLM.register(
        config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration
    )
    AutoModelForCausalLM.register(
        config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration
    )


def prepare_distributed() -> None:
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", device_id=device)


async def evaluate(eval_config: EvalConfig) -> None:
    profile = profiles.get_profile(eval_config.profile)

    result_manager = ResultManager(eval_config.result_base_dir / eval_config.name)
    result_manager.load_model_outputs()

    patch_hf()
    prepare_distributed()

    processor: ProcessorMixin = AutoProcessor.from_pretrained(
        eval_config.model,
        trust_remote_code=True,
    )

    model: _BaseModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        eval_config.model,
        torch_dtype="auto",
        tp_plan="auto",
        trust_remote_code=True,
    )

    @torch.inference_mode()
    async def generate(conversations: List[Conversation]) -> List[str]:
        hf_conversations = await asyncio.gather(
            *[
                convert_conversation_to_hf_format(conversation)
                for conversation in conversations
            ]
        )

        model_inputs = cast(
            BatchFeature,
            processor.apply_chat_template(
                hf_conversations,
                add_generation_prompt=True,
                do_resize=True,
                num_frames=profile.num_video_sample_frames,
                padding=PaddingStrategy.LONGEST,
                return_dict=True,
                return_tensors="pt",
                tokenize=True,
                video_load_backend="opencv",
            ),
        ).to(model.device)

        generated_outputs = model.generate(
            **model_inputs,
            do_sample=False,
        )
        assert isinstance(generated_outputs, torch.LongTensor)

        decoded_outputs: List[str] = processor.post_process_image_text_to_text(
            generated_outputs
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

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory"
    )
    parser.add_argument(
        "--result-base-dir",
        default=DEFAULT_EVAL_RESULT_BASE_DIR,
        help="Evaluation result base directory",
    )

    args = parser.parse_args()

    eval_config = EvalConfig(
        profile=cast(str, args.profile),
        model=cast(str, args.model),
        batch_size=cast(int, args.batch_size),
        dataset_dir=pathlib.Path(cast(str, args.dataset_dir)),
        result_base_dir=pathlib.Path(cast(str, args.result_base_dir)),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

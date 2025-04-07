import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import physgame_benchmark.profiles as profiles
import physgame_benchmark.utils as utils
from physgame_benchmark.conversation import (
    Conversation,
    ImagePillowContentPart,
    TextContentPart,
    VideoContentPart,
)
from physgame_benchmark.dataset import Dataset, DatasetEntry
from physgame_benchmark.result_manager import ModelOutputEntry, ResultManager

DEFAULT_DATASET_DIR = ".dev/PhysGame/PhysGame-Benchmark"
DEFAULT_EVAL_RESULT_BASE_DIR = ".dev/eval"


@dataclass
class EvalConfig:
    profile: str

    model: str
    api_key: Optional[str]
    base_url: Optional[str]

    batch_size: int
    dataset_dir: Path
    result_base_dir: Path

    @property
    def name(self) -> str:
        return f"{self.model.replace('/', '-')}-{self.profile}"


async def convert_conversation_to_openai_format(
    conversation: Conversation,
) -> List[ChatCompletionMessageParam]:
    openai_messages: List[ChatCompletionMessageParam] = []

    for message in conversation:
        openai_contents: List[ChatCompletionContentPartParam] = []

        for content_part in message.content:
            if isinstance(content_part, TextContentPart):
                openai_contents.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=content_part.text,
                    )
                )

            if isinstance(content_part, ImagePillowContentPart):
                openai_contents.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(
                            url=await utils.convert_pil_image_to_base64_url(
                                content_part.image
                            ),
                        ),
                    )
                )

            elif isinstance(content_part, VideoContentPart):
                raise NotImplementedError("Video content part is not supported yet")

        openai_message = cast(
            ChatCompletionMessageParam,
            {
                "role": message.role,
                "content": openai_contents,
            },
        )

        openai_messages.append(openai_message)

    return openai_messages


async def evaluate(eval_config: EvalConfig) -> None:
    profile = profiles.get_profile(eval_config.profile)

    result_manager = ResultManager(eval_config.result_base_dir / eval_config.name)
    result_manager.load_model_outputs()

    client = AsyncOpenAI(
        api_key=eval_config.api_key,
        base_url=eval_config.base_url,
    )

    async def generate(conversations: List[Conversation]) -> List[str]:
        messages_list = await asyncio.gather(
            *[
                convert_conversation_to_openai_format(conversation)
                for conversation in conversations
            ]
        )

        responses = await asyncio.gather(
            *[
                client.chat.completions.create(
                    messages=messages,
                    model=eval_config.model,
                    temperature=0,
                )
                for messages in messages_list
            ]
        )
        return [str(response.choices[0].message.content) for response in responses]

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

    for dataset_entries in tqdm(dataloader):
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
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="Base URL")

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
        api_key=cast(Optional[str], args.api_key),
        base_url=cast(Optional[str], args.base_url),
        batch_size=cast(int, args.batch_size),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        result_base_dir=Path(cast(str, args.result_base_dir)),
    )

    await evaluate(eval_config)


if __name__ == "__main__":
    asyncio.run(main())

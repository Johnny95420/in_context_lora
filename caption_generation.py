# %%
from dotenv import load_dotenv

load_dotenv(".env")
import asyncio
import base64
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import aiofiles
import nest_asyncio
import tqdm
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from vlm_prompt import check_instruction, instruction
from omegaconf import OmegaConf
from dataclasses import dataclass

config = OmegaConf.load("data_clean_config.yaml")


@dataclass
class PromptGenerationConfig:
    do_check = config["do_check"]
    image_data_path = config["concat_data_path"]
    fail_data_path = config["fail_path"]


nest_asyncio.apply()
# %%
MAX_CONCURRENT_REQUESTS = 5
model_name = config["vlm_model"]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def async_encode_image(image_path: str) -> str:
    async with aiofiles.open(image_path, "rb") as image_file:
        image_data = await image_file.read()
    return base64.b64encode(image_data).decode("utf-8")


async def check_image(
    client: AsyncOpenAI,
    base64_image: str,
    instruction: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=400,
        )
        return "false" not in response.choices[0].message.content.lower()


async def get_caption(
    client: AsyncOpenAI,
    base64_image: str,
    instruction: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=400,
        )
        return response.choices[0].message.content


async def save_caption_data(caption_data: Dict[str, Any], output_path: str) -> None:
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(caption_data, ensure_ascii=False, indent=2))


async def process_image(
    image_path: str,
    caption_data: Dict[str, Any],
    client: AsyncOpenAI,
    check_instruction: str,
    instruction: str,
    caption_data_path: str,
    semaphore: asyncio.Semaphore,
    do_check: bool = True,
) -> None:
    name = Path(image_path).name

    # do not process image if it has been done
    if name in caption_data and caption_data[name]:
        return

    base64_image = await async_encode_image(image_path)
    # check the image format is correct or not
    if do_check:
        if not await check_image(client, base64_image, check_instruction, semaphore):
            caption_data[name] = {"pass_check": False, "caption": ""}
            shutil.move(image_path, PromptGenerationConfig.fail_data_path)
            await save_caption_data(caption_data, caption_data_path)
            return

    caption = await get_caption(client, base64_image, instruction, semaphore)
    caption_data[name] = {"pass_check": True, "caption": caption}
    await save_caption_data(caption_data, caption_data_path)


async def main(caption_data_path, image_data_path, do_check: bool = True):
    client = AsyncOpenAI()
    if os.path.isfile(caption_data_path):
        with open(caption_data_path, "r", encoding="utf-8") as f:
            caption_data = json.load(f)
    else:
        caption_data = {}
    images = glob.glob(f"{image_data_path}/**/*.jpg", recursive=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
        process_image(
            img,
            caption_data,
            client,
            check_instruction,
            instruction,
            caption_data_path,
            semaphore,
            do_check,
        )
        for img in images
    ]

    for task in tqdm.as_completed(tasks, total=len(tasks)):
        await task


# %%
if __name__ == "__main__":
    asyncio.run(
        main(
            "description.json",
            PromptGenerationConfig.image_data_path,
            PromptGenerationConfig.do_check,
        )
    )

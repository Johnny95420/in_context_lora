# %%
import pickle as pkl
import glob
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
from diffusers import AutoencoderKL
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from dataset import PreEmbedDataset
from omegaconf import OmegaConf
data_config = OmegaConf.load("model_config.yaml")["data"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3.5-large"
    device: str = "cuda"
    vae_dtype: torch.dtype = torch.float32
    text_encoder_dtype: torch.dtype = torch.bfloat16
    max_sequence_length: int = 512
    clip_max_length: int = 77


# %%
class ImageTextEncoder:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.vae = None
        self.text_encoders = []
        self.tokenizers = []

    def _load_vae(self) -> None:
        """Load and configure the VAE model."""
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="vae",
            ).to(self.device, dtype=self.config.vae_dtype)
            self.vae.requires_grad_(False)
            self.vae_config_shift_factor = self.vae.config.shift_factor
            self.vae_config_scaling_factor = self.vae.config.scaling_factor
            logger.info("VAE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VAE model: {str(e)}")
            raise

    def _load_text_encoders(self) -> None:
        """Load tokenizers and text encoders."""
        try:
            self.tokenizers = [
                CLIPTokenizer.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="tokenizer",
                ),
                CLIPTokenizer.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="tokenizer_2",
                ),
                T5Tokenizer.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="tokenizer_3",
                ),
            ]

            self.text_encoders = [
                CLIPTextModelWithProjection.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    device_map="auto",
                    torch_dtype=self.config.text_encoder_dtype,
                ).to(self.config.text_encoder_dtype),
                CLIPTextModelWithProjection.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="text_encoder_2",
                    device_map="auto",
                    torch_dtype=self.config.text_encoder_dtype,
                ).to(self.config.text_encoder_dtype),
                T5EncoderModel.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="text_encoder_3",
                    device_map="auto",
                    torch_dtype=self.config.text_encoder_dtype,
                ),
            ]
            logger.info("Text encoders and tokenizers loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text encoders: {str(e)}")
            raise

    def _encode_image(self, pixel: torch.Tensor) -> torch.distributions.Distribution:
        """Encode a single image using the VAE."""
        with torch.no_grad():
            pixel = pixel.to(self.device, dtype=self.config.vae_dtype)
            if pixel.dim() == 3:
                pixel = pixel.unsqueeze(0)
            return self.vae.encode(pixel).latent_dist

    def _encode_prompt_with_clip(
        self,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        prompt: List[str],
        text_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using CLIP model."""
        if text_input_ids is None:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided if text_input_ids is None")
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.config.clip_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(
            text_input_ids.to(self.device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        return (
            prompt_embeds.to(dtype=text_encoder.dtype, device=self.device),
            pooled_prompt_embeds,
        )

    def _encode_prompt_with_t5(
        self,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        prompt: List[str],
        text_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode prompt using T5 model."""
        if text_input_ids is None:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided if text_input_ids is None")
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.config.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

        prompt_embeds = text_encoder(text_input_ids.to(self.device))[0]
        return prompt_embeds.to(dtype=text_encoder.dtype, device=self.device)

    def encode_prompt(
        self,
        prompt: str,
        text_input_ids_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using both CLIP and T5 models."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []

        for i, (tokenizer, text_encoder) in enumerate(
            zip(self.tokenizers[:2], self.text_encoders[:2])
        ):
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                text_encoder,
                tokenizer,
                prompt,
                text_input_ids_list[i] if text_input_ids_list else None,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = self._encode_prompt_with_t5(
            self.text_encoders[-1],
            self.tokenizers[-1],
            prompt,
            text_input_ids_list[-1] if text_input_ids_list else None,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        return prompt_embeds, pooled_prompt_embeds

    def save_vae_para(self, name: str = "vae_config"):
        vae_config = {
            "shift": self.vae_config_shift_factor,
            "scale": self.vae_config_scaling_factor,
        }
        with open(f"./{name}.pkl", "wb") as f:
            pkl.dump(vae_config, f)

    def save_negative_emb(self, name: str = "negative_prompt"):
        with torch.no_grad():
            negative_prompt_emb, pooled_negative_prompt_emb = self.encode_prompt("")
        negative_prompt_emb, pooled_negative_prompt_emb = (
            negative_prompt_emb.cpu(),
            pooled_negative_prompt_emb.cpu(),
        )
        negative_prompt = {
            "negative_prompt_emb": negative_prompt_emb,
            "pooled_negative_prompt_emb": pooled_negative_prompt_emb,
        }
        with open(f"./{name}.pkl", "wb") as f:
            pkl.dump(negative_prompt, f)

    def process_dataset(self, image_paths: List[str]) -> Dict:
        """Process the dataset and generate embeddings."""
        encode_data = {}
        pre_emb_dataset = PreEmbedDataset(image_paths)

        # Process images
        self._load_vae()
        self.save_vae_para()
        for name, pixel, _ in tqdm.tqdm(pre_emb_dataset, desc="Encoding images"):
            encode_data[name] = {
                "latent_dist": self._encode_image(pixel),
                "pool_text_emb": None,
                "text_emb": None,
            }
        del self.vae
        torch.cuda.empty_cache()

        # Process text
        self._load_text_encoders()
        self.save_negative_emb()
        for name, _, text in tqdm.tqdm(pre_emb_dataset, desc="Encoding text"):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(text)
                encode_data[name]["pool_text_emb"] = pooled_prompt_embeds[0].cpu()
                encode_data[name]["text_emb"] = prompt_embeds[0].cpu()
        del self.text_encoders
        del self.tokenizers
        torch.cuda.empty_cache()
        return encode_data


# %%
if __name__ == "__main__":
    config = ModelConfig()
    encoder = ImageTextEncoder(config)
    try:
        image_paths = glob.glob(f"{data_config['data_path']}/**/*.jpg", recursive=True)
        if not image_paths:
            raise ValueError("No images found in the specified directory")

        encode_data = encoder.process_dataset(image_paths)
        logger.info("Dataset processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")

    for key in tqdm.tqdm(encode_data):
        with open(f"{data_config["embedded_data_path"]}/{key}.pkl", "wb") as f:
            pkl.dump(encode_data[key], f)

# %%

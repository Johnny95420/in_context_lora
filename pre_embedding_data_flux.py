# %%
import glob
import logging
import pickle as pkl
from dataclasses import dataclass

import torch
import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from pre_embedding_data import ImageTextEncoder
from omegaconf import OmegaConf
data_config = OmegaConf.load("model_config.yaml")["data"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    device: str = "cuda"
    vae_dtype: torch.dtype = torch.float32
    text_encoder_dtype: torch.dtype = torch.bfloat16
    max_sequence_length: int = 512
    clip_max_length: int = 77


class ImageTextEncoderFlux(ImageTextEncoder):
    def __init__(self, config):
        super().__init__(config)

    def _load_text_encoders(self):
        """Load tokenizers and text encoders."""
        try:
            self.tokenizers = [
                CLIPTokenizer.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="tokenizer",
                ),
                T5Tokenizer.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="tokenizer_2",
                ),
            ]

            self.text_encoders = [
                CLIPTextModel.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    device_map="auto",
                    torch_dtype=self.config.text_encoder_dtype,
                ).to(self.config.text_encoder_dtype),
                T5EncoderModel.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    subfolder="text_encoder_2",
                    device_map="auto",
                    torch_dtype=self.config.text_encoder_dtype,
                ),
            ]
            logger.info("Text encoders and tokenizers loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text encoders: {str(e)}")
            raise

    def save_vae_para(self, name="vae_config_flux"):
        vae_config = {
            "shift": self.vae_config_shift_factor,
            "scale": self.vae_config_scaling_factor,
            "block_channels": self.vae.config.block_out_channels,
        }
        with open(f"./{name}.pkl", "wb") as f:
            pkl.dump(vae_config, f)

    def save_negative_emb(self, name="negative_prompt_flux"):
        return super().save_negative_emb(name)

    def _encode_prompt_with_clip(
        self, text_encoder, tokenizer, prompt, text_input_ids=None
    ):
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
        pooled_prompt_embeds = prompt_embeds.pooler_output
        return pooled_prompt_embeds

    def encode_prompt(self, prompt, text_input_ids_list=None):
        """Encode prompt using both CLIP and T5 models."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            self.text_encoders[0],
            self.tokenizers[0],
            prompt,
            text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = self._encode_prompt_with_t5(
            self.text_encoders[-1],
            self.tokenizers[-1],
            prompt,
            text_input_ids_list[-1] if text_input_ids_list else None,
        )
        return prompt_embeds, pooled_prompt_embeds


# %%
if __name__ == "__main__":
    config = ModelConfig()
    encoder = ImageTextEncoderFlux(config)
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

# %%
import copy
import glob
import logging
import os
import pickle as pkl
import shutil
from functools import partial
from random import shuffle

import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.training_utils import EMAModel

set_seed(42)
from diffusers import (
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbeddedDataset
from train_script import is_compiled_module
from omegaconf import OmegaConf

config = OmegaConf.load("model_config.yaml")
# %%
logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# %%
tracker_name = config["model"]["tracker_name"]
lr_warmup_steps = config["model"]["lr_warmup_steps"]
max_train_steps = config["model"]["max_train_steps"]
pretrained_model_name_or_path = config["model"]["pretrained_model_name_or_path"]
gradient_accumulation_steps = config["model"]["gradient_accumulation_steps"]
learning_rate = config["model"]["learning_rate"]
batch_size = config["model"]["batch_size"]
output_dir = config["model"]["output_dir"]
logging_dir = config["model"]["logging_dir"]
upcast_before_saving = config["model"]["upcast_before_saving"]
checkpoints_total_limit = config["model"]["checkpoints_total_limit"]
vae_config_path = config["model"]["vae_config_path"]
negative_prompt_emb_path = config["model"]["negative_prompt_emb_path"]


# %%
def unwrap_model(model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        transformer_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None

        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                if upcast_before_saving:
                    model = model.to(torch.float32)
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        FluxPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
        )


def load_model_hook(models, input_dir):
    transformer_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(unwrap_model(transformer))):
            transformer_ = model
        else:
            raise ValueError(f"unexpected save model: {model.__class__}")

    lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

    transformer_state_dict = {
        f"{k.replace('transformer.', '')}": v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(
        transformer_, transformer_state_dict, adapter_name="default"
    )
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )


def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def step(
    data,
    transformer,
    noise_scheduler_copy,
    vae_config_shift_factor,
    vae_config_scaling_factor,
    guidace: float = 1,
):
    data = {key: data[key].to(accelerator.device) for key in data}
    model_input, pooled_prompt_embeds, prompt_embeds = (
        data["latent_dist"],
        data["pool_text_emb"],
        data["text_emb"],
    )
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
        device=accelerator.device, dtype=torch.bfloat16
    )
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=torch.bfloat16)
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0],
        model_input.shape[2] // 2,
        model_input.shape[3] // 2,
        accelerator.device,
        torch.bfloat16,
    )
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=bsz,
        logit_mean=0.0,
        logit_std=1.0,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    packed_noisy_model_input = FluxPipeline._pack_latents(
        noisy_model_input,
        batch_size=model_input.shape[0],
        num_channels_latents=model_input.shape[1],
        height=model_input.shape[2],
        width=model_input.shape[3],
    )
    guidance = torch.tensor([guidace], device=accelerator.device)
    guidance = guidance.expand(model_input.shape[0])
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    vae_scale_factor = 2 ** (len(block_channels) - 1)
    model_pred = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    model_pred = model_pred * (-sigmas) + noisy_model_input
    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme="logit_normal", sigmas=sigmas
    )
    target = model_input
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        ),
        1,
    )
    loss = loss.mean()
    return loss


def compute_grad_norm(transformer):
    total_norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), 2)
                for p in accelerator.unwrap_model(transformer).parameters()
                if p.grad is not None
            ]
        )
    )
    return total_norm


def validation_generation(
    prompt_embeds,
    pooled_prompt_embeds,
    global_step,
    logging_dir,
):
    if (
        accelerator.is_main_process
        or accelerator.distributed_type == DistributedType.DEEPSPEED
    ):
        pipe = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
        )
        pipe.scheduler.set_shift(7)
        with torch.autocast("cuda:0"):
            image = pipe(
                guidance_scale=3.5,
                num_inference_steps=25,
                prompt_embeds=prompt_embeds.to("cuda"),
                pooled_prompt_embeds=pooled_prompt_embeds.to("cuda"),
                width=1120,
                height=560,
            )
        del pipe
        image.images[0].save(f"{logging_dir}/img_{global_step}.jpg")
        torch.cuda.empty_cache()
        return image.images[0]


def save_ckpt(accelerator, global_step):
    if (
        accelerator.is_main_process
        or accelerator.distributed_type == DistributedType.DEEPSPEED
    ):
        if checkpoints_total_limit is not None:
            checkpoints = os.listdir(output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= checkpoints_total_limit:
                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        ema_save_path = os.path.join(
            output_dir, f"checkpoint-{global_step}", f"ema_state_dict-{global_step}.pt"
        )
        torch.save(ema_transformer.state_dict(), ema_save_path)
        logger.info(f"Saved state to {save_path}")


def guidance_embed_bypass_forward(self, timestep, guidance, pooled_projection):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype)
    )  # (N, D)
    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = timesteps_emb + pooled_projections
    return conditioning


def bypass_flux_guidance(transformer):
    if hasattr(transformer.time_text_embed, "_bfg_orig_forward"):
        return
    # dont bypass if it doesnt have the guidance embedding
    if not hasattr(transformer.time_text_embed, "guidance_embedder"):
        return
    transformer.time_text_embed._bfg_orig_forward = transformer.time_text_embed.forward
    transformer.time_text_embed.forward = partial(
        guidance_embed_bypass_forward, transformer.time_text_embed
    )


def restore_flux_guidance(transformer):
    if not hasattr(transformer.time_text_embed, "_bfg_orig_forward"):
        return
    transformer.time_text_embed.forward = transformer.time_text_embed._bfg_orig_forward
    del transformer.time_text_embed._bfg_orig_forward


# %%
if __name__ == "__main__":
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers(tracker_name)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    # %%
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1
    )
    transformer.add_adapter(transformer_lora_config)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": learning_rate,
    }

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    ema_transformer = None
    if accelerator.is_main_process:
        # Initialize EMA model with the transformer parameters
        ema_transformer = EMAModel(
            parameters=transformer_lora_parameters,
            model_cls=type(unwrap_model(transformer)),
            model_config=transformer.config,
            use_accelerator=True,
            decay=0.99,
            foreach=True,
        )
        # Prepare EMA model with Accelerator
        ema_transformer = accelerator.prepare(ema_transformer)
    # %%
    with open(vae_config_path, "rb") as f:
        vae_config = pkl.load(f)
    vae_config_shift_factor, vae_config_scaling_factor, block_channels = (
        vae_config["shift"],
        vae_config["scale"],
        vae_config["block_channels"],
    )

    with open(negative_prompt_emb_path, "rb") as f:
        negative_prompt = pkl.load(f)
    negative_prompt_emb, pooled_negative_prompt_emb = (
        negative_prompt["negative_prompt_emb"],
        negative_prompt["pooled_negative_prompt_emb"],
    )

    files = glob.glob(f"{config['data']['embedded_data_path']}/*.*")
    shuffle(files)
    l = len(files)
    train_l = int(l * 0.9)
    train_files, val_files = files[:train_l], files[train_l:]
    train_dataset = EmbeddedDataset(train_files)
    val_dataset = EmbeddedDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW([transformer_parameters_with_lr])
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=1,
    )
    transformer, optimizer, train_loader, lr_scheduler, val_loader = (
        accelerator.prepare(
            transformer, optimizer, train_loader, lr_scheduler, val_loader
        )
    )
    # %%
    global_step = 0
    guidance_set = np.linspace(1, 2, max_train_steps)
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    checkpointing_steps = 250
    validation_steps = 250
    while global_step <= max_train_steps:
        for data in train_loader:
            transformer.train()
            with accelerator.accumulate([transformer]):
                guidance = guidance_set[global_step]
                if np.random.binomial(1, 0.05) == 1:
                    data["text_emb"] = negative_prompt_emb.repeat([batch_size, 1, 1])
                    data["pool_text_emb"] = pooled_negative_prompt_emb.repeat(
                        [batch_size, 1]
                    )
                    guidance = 1
                loss = step(
                    data,
                    transformer,
                    noise_scheduler_copy,
                    vae_config_shift_factor,
                    vae_config_scaling_factor,
                    guidace=guidance,
                )
                accelerator.backward(loss)
                grad_norm = compute_grad_norm(transformer)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, 3)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.sync_gradients and ema_transformer is not None:
                    ema_transformer.step(transformer_lora_parameters)

                if global_step % checkpointing_steps == 0:
                    save_ckpt(accelerator, global_step)

                if global_step % validation_steps == 0:
                    transformer.eval()
                    ema_transformer.store(transformer_lora_parameters)
                    ema_transformer.copy_to(transformer_lora_parameters)
                    total_val_loss = 0
                    with torch.no_grad():
                        for data in val_loader:
                            loss = step(
                                data,
                                transformer,
                                noise_scheduler_copy,
                                vae_config_shift_factor,
                                vae_config_scaling_factor,
                                3.5,
                            )
                            total_val_loss += loss / len(val_loader)
                        torch.cuda.empty_cache()
                        validation_generation(
                            data["text_emb"][[0]],
                            data["pool_text_emb"][[0]],
                            global_step,
                            logging_dir,
                        )
                    logs = {
                        "train_loss": loss.detach().item(),
                        "val_loss": total_val_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "gradient_norm": grad_norm.item(),
                    }
                    ema_transformer.restore(transformer_lora_parameters)
                else:
                    logs = {
                        "train_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "gradient_norm": grad_norm.item(),
                    }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

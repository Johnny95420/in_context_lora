# %%
from typing import List, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from diffusers import BitsAndBytesConfig, FluxImg2ImgPipeline, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_control_img2img import (
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.training_utils import EMAModel
from ipycanvas import Canvas
from IPython.display import display
from PIL import Image, ImageDraw

pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
lora_path = "/ic_lora/output_flux/checkpoint-9000"
ema_weight_path = lora_path + "/ema_state_dict-9000.pt"


class PolygonMaskAnnotator:
    def __init__(self, img):
        self.img = img
        self.img_np = np.array(self.img)
        self.h, self.w = self.img_np.shape[:2]

        self.canvas = Canvas(width=self.w, height=self.h, sync_image_data=True)
        self.canvas.put_image_data(self.img_np, 0, 0)
        self.polygon_points = []
        self.mask_preview = None
        self.polygon_mask_var = None
        self.output = widgets.Output()

        self.button_demo = widgets.Button(description="Demo Mask")
        self.button_save = widgets.Button(description="Save Mask")
        self.button_clear = widgets.Button(description="Clear")
        self.btns = widgets.HBox(
            [self.button_demo, self.button_save, self.button_clear]
        )

        self.canvas.on_mouse_down(self._on_mouse_down)
        self.button_demo.on_click(self._on_demo_clicked)
        self.button_save.on_click(self._on_save_clicked)
        self.button_clear.on_click(self._on_clear_clicked)

    def _draw_polygon(self, show_mask=False, mask=None):
        self.canvas.clear()
        self.canvas.put_image_data(self.img_np, 0, 0)
        if show_mask and mask is not None:
            self.canvas.global_alpha = 0.5
            mask_rgba = np.zeros((self.h, self.w, 4), dtype=np.uint8)
            mask_rgba[..., 1] = 255
            mask_rgba[..., 3] = mask
            self.canvas.put_image_data(mask_rgba, 0, 0)
            self.canvas.global_alpha = 1.0

        if self.polygon_points:
            self.canvas.stroke_style = "lime"
            self.canvas.line_width = 2
            self.canvas.stroke_polygon(self.polygon_points)
            for x, y in self.polygon_points:
                self.canvas.fill_style = "red"
                self.canvas.fill_circle(x, y, 4)

    def _on_mouse_down(self, x, y):
        self.polygon_points.append((int(x), int(y)))
        self._draw_polygon()

    def _make_mask(self, points):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        if len(points) > 2:
            img_pil = Image.fromarray(mask)
            ImageDraw.Draw(img_pil).polygon(points, outline=255, fill=255)
            return np.array(img_pil)
        else:
            return mask

    def _on_demo_clicked(self, b):
        with self.output:
            self.output.clear_output()
            if len(self.polygon_points) < 3:
                print("Please at least draw 3 points")
                self._draw_polygon()
            else:
                self.mask_preview = self._make_mask(self.polygon_points)
                self._draw_polygon(show_mask=True, mask=self.mask_preview)
                print("Demo polygon mask (unsaved)")

    def _on_save_clicked(self, b):
        with self.output:
            self.output.clear_output()
            if self.mask_preview is not None:
                self.polygon_mask_var = self.mask_preview.copy()
                print("Saved !")
            else:
                print("Please demo at first")

    def _on_clear_clicked(self, b):
        with self.output:
            self.output.clear_output()
        self.polygon_points.clear()
        self.mask_preview = None
        self._draw_polygon()

    def show(self):
        display(self.canvas, self.btns, self.output)

    def get_mask(self):
        if self.polygon_mask_var is None:
            print("Please draw a polygon and save at first!")
        return self.polygon_mask_var


# %%
def get_conditional_latents(
    sample: torch.Tensor,
    latent_timestep: torch.Tensor,
    shape: Tuple,
    device: str,
    dtype: str,
    generator: Optional[torch.Generator],
):
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    sigmas = pipe.scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)
    schedule_timesteps = pipe.scheduler.timesteps.to(sample.device, dtype=torch.float32)
    timestep = latent_timestep.to(sample.device, dtype=torch.float32)
    step_indices = [
        pipe.scheduler.index_for_timestep(t, schedule_timesteps) for t in timestep
    ]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(sample.shape):
        sigma = sigma.unsqueeze(-1)
    condition_latent = (1.0 - sigma) * sample + sigma * noise
    condition_latent = torch.cat([condition_latent, sample], axis=0)
    return condition_latent


@torch.no_grad
def encode_image(
    pipe: FluxImg2ImgPipeline,
    batch_size: int,
    init_image: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    image_latents = (
        pipe._encode_vae_image(image=init_image, generator=generator)
        if init_image.shape[1] != pipe.latent_channels
        else init_image
    )

    # Adjust batch size for latents
    if batch_size > image_latents.shape[0]:
        if batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat(
                [image_latents] * additional_image_per_prompt, dim=0
            )
        else:
            raise ValueError(
                f"Cannot duplicate image of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
    else:
        image_latents = torch.cat([image_latents], dim=0)
    return image_latents


@torch.no_grad
def process_image_pipeline(
    pipe: FluxImg2ImgPipeline,
    mask: torch.Tensor,
    prompt: str = "",
    prompt_2: Optional[str] = None,
    image: Image = None,
    height: int = 592,
    width: int = 1200,
    strength: float = 0.7,
    num_inference_steps: int = 25,
    guidance_scale: float = 6.5,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    lora_scale: float = 1,
    sigmas: List = None,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.tensor] = None,
    shift: float = 10,
    device: str = "cuda",
):
    """
    Process an image through the diffusion pipeline.

    Args:
        pipe: The diffusion pipeline object
        prompt: Text prompt for image generation
        image: Input image tensor
        height: Output image height
        width: Output image width
        strength: Strength parameter for image transformation
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for the model
        num_images_per_prompt: Number of images per prompt
        max_sequence_length: Maximum sequence length for prompt encoding
        lora_scale: LoRA scale parameter
        device: Device to run the computation on

    Returns:
        Processed image tensor
    """
    # Initialize parameters
    if prompt_2 is None:
        prompt_2 = prompt
    negative_prompt = negative_prompt_2 = None
    prompt_embeds = negative_prompt_embeds = pooled_prompt_embeds = (
        negative_pooled_prompt_embeds
    ) = None

    # Set pipeline parameters
    pipe.scheduler.set_shift(shift)
    pipe._guidance_scale = guidance_scale
    pipe._joint_attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    # Validate inputs
    pipe.check_inputs(
        prompt,
        prompt_2,
        strength,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=max_sequence_length,
    )

    # Preprocess image
    init_image = pipe.image_processor.preprocess(image, height=height, width=width)
    init_image = init_image.to(dtype=torch.float32)

    # Determine batch size
    batch_size = (
        1
        if isinstance(prompt, str)
        else len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
    )

    # Encode prompt
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # Setup timesteps and sigmas
    sigmas = (
        np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if sigmas is None
        else sigmas
    )
    image_seq_len = (height // pipe.vae_scale_factor // 2) * (
        width // pipe.vae_scale_factor // 2
    )
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps, strength, device
    )

    if num_inference_steps < 1:
        raise ValueError(
            f"After adjusting num_inference_steps by strength parameter: {strength}, "
            f"the number of pipeline steps is {num_inference_steps} which is < 1."
        )

    # Encode initial image
    dtype = prompt_embeds.dtype
    init_image = init_image.to(device=device, dtype=dtype)
    image_latents = encode_image(pipe, batch_size, init_image, generator)

    # Generate noise and condition latent
    num_channels_latents = pipe.transformer.config.in_channels // 4
    vae_height = 2 * (height // (pipe.vae_scale_factor * 2))
    vae_width = 2 * (width // (pipe.vae_scale_factor * 2))
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size, vae_height // 2, vae_width // 2, device, dtype
    )
    condition_latent = get_conditional_latents(
        sample=image_latents,
        latent_timestep=timesteps.repeat(batch_size * num_images_per_prompt),
        shape=(batch_size, num_channels_latents, vae_height, vae_width),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # Process diffusion steps
    latents = condition_latent[[0]]
    guidance = (
        torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(
            latents.shape[0]
        )
        if pipe.transformer.config.guidance_embeds
        else None
    )

    for i, t in tqdm.tqdm(enumerate(timesteps)):
        latents = pipe._pack_latents(
            latents, batch_size, num_channels_latents, vae_height, vae_width
        )
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        latents[:, :, mask] = condition_latent[i + 1, :, mask]

    # Decode final image
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image)


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

pipe = FluxImg2ImgPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
pipe.text_encoder_2 = pipe.text_encoder_2.to(torch.bfloat16)

pipe.load_lora_weights(
    lora_path,
    weight_name="pytorch_lora_weights.safetensors",
)
transformer_lora_parameters = list(
    filter(lambda p: p.requires_grad, pipe.transformer.parameters())
)
ema_transformer = EMAModel(
    parameters=transformer_lora_parameters,
    model_cls=type(pipe.transformer),
    model_config=transformer.config,
    use_accelerator=True,
    decay=0.99,
    foreach=True,
)
ema_transformer.load_state_dict(torch.load(ema_weight_path))
ema_transformer.store(transformer_lora_parameters)
ema_transformer.copy_to(transformer_lora_parameters)
pipe.transformer.eval()
# %%
if __name__ == "__main__":
    prompt = """This <Brand> bralette is a delicate, luxurious lingerie piece designed for elegance and comfort, featuring soft beige fabric with intricate floral embroidery and fine metallic trim along its edges, ideal for intimate wear or layering under light summer attire.\n\n
    <IMAGE 1> The front view of the <Brand> bralette displays triangle cups with semi-sheer floral embroidery on beige fabric, bordered by shimmering gunmetal metallic trim that outlines the shape and adds subtle glamour. Thin, adjustable straps rise from the cups, enhancing the minimalistic, lightweight silhouette. The texture is soft and slightly translucent, blending delicate craft with a sleek, refined edge.\n\n
    <IMAGE 2> From the back, the <Brand> bralette reveals the smooth, solid beige inner fabric of the cups and the adjustable hook clasp closure at the center. The metallic trim continues around the band, maintaining the piece’s cohesive shimmer. The thin straps maintain simplicity, emphasizing clean lines and functionality alongside the decorative front.\n\n
    <IMAGE 3> The model, a slender young woman with olive skin and curly brunette hair styled loosely in a messy bun, is captured from behind wearing the <Brand> bralette paired with a skinny short jeans. The skinny short jeans fit her hips and thighs perfectly. She accessorizes with black leather gloves and black ankle boots, blending casual ease with an edgy vibe. The scene is lit by dim daylight against a pristine white wall, emphasizing a clean, minimalist, and modern mood.
    """

    obj1 = Image.open("/ic_lora/test_images/obj1.jpg")
    obj2 = Image.open("/ic_lora/test_images/obj2.jpg")
    human = Image.open("/ic_lora/test_images/model_edit_3.jpg")
    obj1 = obj1.resize([400, 600])
    obj2 = obj2.resize([400, 600])
    human = human.resize([400, 600])
    img = Image.fromarray(
        np.concatenate([np.array(obj1), np.array(obj2), np.array(human)], axis=1)
    )
    annotator = PolygonMaskAnnotator(img)
    annotator.show()
    # %%
    # * the selected part need to be re-generating
    mask = annotator.get_mask()
    mask = mask < 127

    w, h = img.size
    vae_w = 2 * (w // (pipe.vae_scale_factor * 2))
    vae_h = 2 * (h // (pipe.vae_scale_factor * 2))
    mask = Image.fromarray(mask).resize([vae_w, vae_h])
    mask = np.array(mask)
    mask[:, :100] = 1
    plt.figure(figsize=(16, 16))
    plt.imshow(np.array(img.resize([vae_w, vae_h])))
    plt.imshow(mask, cmap="gray", alpha=0.5)
    # %%
    pipe.set_adapters(["default_0"], adapter_weights=[1.0])
    images = process_image_pipeline(
        pipe=pipe,
        mask=mask,
        prompt=prompt,
        image=img,
        guidance_scale=7,
        shift=10,
        strength=0.65,
        num_inference_steps=25,
    )
    images[0]
    # %%

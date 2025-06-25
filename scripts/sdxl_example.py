import torch
import os

# 设置Hugging Face缓存目录到用户主目录，避免scratch空间配额问题
os.environ['HF_HOME'] = '/u/lanius/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/u/lanius/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/u/lanius/huggingface'

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, parallelism="naive_patch")
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    mode="sync_gn",
    split_scheme="row",
    use_safetensors=True,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if distri_config.rank == 0:
    image.save("astronaut.png")

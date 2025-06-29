from datasets import load_dataset
from dateutil import parser
from datetime import datetime
import torch
import time

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
import os
import torch.distributed as dist

import pandas as pd
import matplotlib.pyplot as plt

os.environ['HF_HOME'] = '/u/lanius/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/u/lanius/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/u/lanius/huggingface'

ds = load_dataset(
    'poloclub/diffusiondb',
    'large_text_only',
    split='train',
    streaming=True
)

last_prompt = None
prompt_ts = []
for rec in ds:
    p = rec['prompt']
    if p == last_prompt:
        continue
    last_prompt = p

    ts = rec['timestamp']
    if not isinstance(ts, datetime):
        ts = parser.isoparse(ts)
    prompt_ts.append((p, ts))

    if len(prompt_ts) >= 20:
        break

prompt_ts.sort(key=lambda x: x[1])

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, use_cuda_graph=False, mode="full_sync")
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16"
    )

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

if distri_config.rank == 0:
    print(f"Starting generation of {len(prompt_ts)} images...")
    generation_start_time = time.time()

for idx, (prompt, _) in enumerate(prompt_ts):
    gen = torch.Generator(device="cuda").manual_seed(233 + idx)
    image = pipeline(prompt=prompt, generator=gen).images[0]

    if distri_config.rank == 0:
        filename = f"image_{idx:03d}.png"
        image.save(filename)
        print(f"Saved {filename} ({idx+1}/{len(prompt_ts)})")


if distri_config.rank == 0:
    generation_end_time = time.time()
    total_time = generation_end_time - generation_start_time
    avg_time_per_image = total_time / len(prompt_ts)

    print(f"\n=== Generation Complete Statistics ===")
    print(f"Total images: {len(prompt_ts)}")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Throughput: {len(prompt_ts)/total_time:.2f} images/second")

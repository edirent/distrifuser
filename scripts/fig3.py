from datasets import load_dataset
from dateutil import parser
from datetime import datetime
import torch
import time

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
import os

os.environ['HF_HOME'] = '/u/lanius/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/u/lanius/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/u/lanius/huggingface'

# Load metadata stream, extract prompt and timestamp, deduplicate, collect 100 entries
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
    # Convert to datetime uniformly
    if not isinstance(ts, datetime):
        ts = parser.isoparse(ts)
    prompt_ts.append((p, ts))

    if len(prompt_ts) >= 100:
        break

# Sort by timestamp in ascending order
prompt_ts.sort(key=lambda x: x[1])

# Initialize and load DistriSDXLPipeline
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    mode="sync_gn",
    split_scheme="row",
)
# Disable progress bar for non-main processes
pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

# Start timing - record image generation start time
if distri_config.rank == 0:
    print(f"Starting generation of {len(prompt_ts)} images...")
    generation_start_time = time.time()

# Generate and save images in loop
for idx, (prompt, _) in enumerate(prompt_ts):
    # Use different seeds for diversity
    gen = torch.Generator(device="cuda").manual_seed(233 + idx)
    image = pipeline(prompt=prompt, generator=gen).images[0]
    
    # Only rank 0 process saves files to avoid duplication
    if distri_config.rank == 0:
        filename = f"image_{idx:03d}.png"
        image.save(filename)
        print(f"Saved {filename} ({idx+1}/{len(prompt_ts)})")

# End timing and output total time
if distri_config.rank == 0:
    generation_end_time = time.time()
    total_time = generation_end_time - generation_start_time
    avg_time_per_image = total_time / len(prompt_ts)
    
    print(f"\n=== Generation Complete Statistics ===")
    print(f"Total images: {len(prompt_ts)}")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"Throughput: {len(prompt_ts)/total_time:.2f} images/second")

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

    if len(prompt_ts) >= 2:
        break

# Sort by timestamp in ascending order
prompt_ts.sort(key=lambda x: x[1])

# Initialize and load DistriSDXLPipeline
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, parallelism="naive_patch", use_cuda_graph=False)
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
    
unet = pipeline.pipeline.unet

dist.barrier()

all_data = [None] * distri_config.world_size
dist.gather_object(unet.times, all_data if dist.get_rank() == 0 else None, dst=0)

if dist.get_rank() == 0:
    step_plot = [3, 4]
    plot_data = []
    for rank, rank_data in enumerate(all_data):
        if not rank_data:
            continue
        for step_data in rank_data:
            if step_data['step'] in step_plot:
                comp_duration = step_data['comp_duration']
                comm_duration = step_data['comm_duration']
                comp_start = 0
                comm_start = comm_duration
                plot_data.append({
                    "Rank": rank,
                    "Step": step_data['step'],
                    "Task": "Computation",
                    "Start": comp_start,
                    "Duration": comp_duration,
                })
                plot_data.append({
                    "Rank": rank,
                    "Step": step_data['step'],
                    "Task": "Communication",
                    "Start": comm_start,
                    "Duration": comm_duration,
                })
    df = pd.DataFrame(plot_data)
    
    def plot_gantt(dataframe, steps):
        num_ranks = dataframe['Rank'].nunique()
        fig, axes = plt.subplots(len(steps), 1, figsize=(15, 4 * len(steps)), squeeze=False)
        fig.suptitle('GPU Computation and Communication Gantt Chart', fontsize=16)

        colors = {"Computation": "skyblue", "Communication": "salmon"}

        for i, step in enumerate(steps):
            ax = axes[i, 0]
            step_df = dataframe[dataframe['Step'] == step]
            
            for rank in range(num_ranks):
                rank_df = step_df[step_df['Rank'] == rank]
                for _, row in rank_df.iterrows():
                    ax.barh(y=f"GPU {row['Rank']}", width=row['Duration'], left=row['Start'], 
                            height=0.6, color=colors[row['Task']], edgecolor='black')

            ax.set_xlabel("Time (ms)")
            ax.set_title(f"Timeline for Step {step}")
            ax.invert_yaxis() # 让 GPU 0 在最上面
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            # 创建图例
            patches = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
            ax.legend(patches, colors.keys())
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存或显示图表
        output_filename = "gantt_chart.png"
        plt.savefig(output_filename)
        print(f"\nGantt chart saved to {output_filename}")

    if not df.empty:
        plot_gantt(df, steps_to_plot)
    else:
        print("\nNo timing data found for the selected steps to plot.")


# 清理分布式环境
if dist.is_initialized():
    dist.destroy_process_group()
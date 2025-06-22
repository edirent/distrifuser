from datasets import load_dataset
from dateutil import parser
from datetime import datetime
import torch

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

# 1. 流式加载 metadata，只取 prompt 和 timestamp，去重，收集 100 条
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
    # 统一转换为 datetime
    if not isinstance(ts, datetime):
        ts = parser.isoparse(ts)
    prompt_ts.append((p, ts))

    if len(prompt_ts) >= 100:
        break

# 2. 按 timestamp 升序排序
prompt_ts.sort(key=lambda x: x[1])

# 3. 初始化并加载 DistriSDXLPipeline
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
)
# 屏蔽非主进程的进度条
pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

# 4. 循环生成并保存图片
for idx, (prompt, _) in enumerate(prompt_ts):
    # 不同 seed 保证多样性
    gen = torch.Generator(device="cuda").manual_seed(233 + idx)
    image = pipeline(prompt=prompt, generator=gen).images[0]
    
    # 只由 rank 0 进程保存文件，避免重复
    if distri_config.rank == 0:
        filename = f"image_{idx:03d}.png"
        image.save(filename)
        print(f"Saved {filename}")

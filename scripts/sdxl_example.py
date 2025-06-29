import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(
    height=1024,
    width=1024,
    warmup_steps=2,
    mode="full_sync",  # Use full_sync mode for this exampl
    use_cuda_graph=False,
)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16"
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(1 + 1 + 3):
        if prof.step_num >= 1:  # skip wait
            with record_function("distri_pipeline_step"):
                image = pipeline(
                    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                    generator=torch.Generator(device="cuda").manual_seed(233),
                ).images[0]
        prof.step()

if distri_config.rank == 0:
    image.save("astronaut.png") 

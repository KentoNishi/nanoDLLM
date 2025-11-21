# nanoDLLM

Lightweight Block Diffusion language model with a minimal training loop and evaluation utilities.

## Layout
- `model.py`: Block Diffusion model definition.
- `train.py`: training script for the diffusion model (FlexAttention + Block Diffusion objective).
- `requirements.txt`: core dependencies.

## Quickstart
```bash
pip install -r requirements.txt
# train
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

### Data options
- `dataset_mode=fineweb` (default): trains on FineWebEdu binary shards at `data/finewebedu10B/*`.
- `dataset_mode=combined|combined_guidance|cbt|easymath`: uses the course CBT/EasyMath datasets provided by the `conditional_dllm_class_project` package (install from `cs2420_cs2823r_final_project/` via `pip install -e cs2420_cs2823r_final_project`). `combined_guidance` injects a two-class guidance signal for conditioning the diffusion model; set via `DATASET_MODE=combined_guidance` (or edit `Hyperparameters`).
- For guided evaluation, run `python cs2420_cs2823r_final_project/eval/eval_diffusion.py --use-guidance ...` to load the guidance-aware dataset and checkpoints.

## Notes
Block Diffusion masks tokens per block, applies block-aware attention, and trains with a masked cross-entropy objective weighted by noise level. See `model.py` for details.

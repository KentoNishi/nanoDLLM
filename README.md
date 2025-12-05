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

### Training workflow
1. **Pretrain on FineWebEdu10B**
   ```bash
   torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
     train.py --dataset_mode=fineweb --run_id fineweb-base
   ```
   - Checkpoints land in `logs/fineweb-base/state_latest.pt`.
   - Killing/restarting the same command automatically resumes from `state_latest.pt`.

2. **Finetune on the combined CBT/EasyMath dataset**
   ```bash
   cd path/to/conditional_dllm_class_project
   pip install -e .
   cd -
   torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
     train.py --dataset_mode=combined --run_id combined-ft \
     --pretrained_checkpoint logs/fineweb-base/state_latest.pt
   ```
   - For `dataset_mode=combined|cbt|easymath`, the script loads the specified pretrained checkpoint by default (fallbacks to `logs/fineweb-base/state_latest.pt` when available) and starts new optimizer states for transfer learning.

3. **Conditional finetune**
   ```bash
   torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
     train.py --dataset_mode=combined_guidance --run_id combined-guidance \
     --pretrained_checkpoint logs/fineweb-base/state_latest.pt
   ```
   - Adds a two-class guidance embedding (CBT vs EasyMath) and trains the conditional DLLM variant.

### Data options
- `dataset_mode=fineweb` (default): trains on FineWebEdu shards at `data/finewebedu10B/*`.
- `dataset_mode=combined|combined_guidance|cbt|easymath`: uses the CBT/EasyMath datasets provided by the `conditional_dllm_class_project` package (install by running `pip install -e .` inside that repo directory). `combined_guidance` injects a two-class guidance signal for conditioning the diffusion model.
- For guided evaluation, run `python conditional_dllm_class_project/eval/eval_diffusion.py --use-guidance ...` with checkpoints produced by the conditional run.

## Notes
Block Diffusion masks tokens per block, applies block-aware attention, and trains with a masked cross-entropy objective weighted by noise level. See `model.py` for details.

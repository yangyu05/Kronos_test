# Running Kronos Finetuning with GPU

This guide explains how to run the finetuning scripts on a GPU-enabled machine (AWS EC2, local GPU, etc.).

## Prerequisites

1. **GPU-enabled machine** with CUDA support
2. **CUDA drivers** installed
3. **PyTorch with CUDA** support installed

## Step 1: Verify GPU Availability

First, verify that your GPU is detected:

```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Or use nvidia-smi (if NVIDIA drivers are installed)
nvidia-smi
```

Expected output:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA A10G (or your GPU name)
```

## Step 2: Verify Data is Ready

Ensure your processed datasets exist:

```bash
ls -lh data/processed_datasets/*.pkl
```

You should see:
- `train_data.pkl`
- `val_data.pkl`
- `test_data.pkl`

## Step 3: Run Tokenizer Finetuning

### Single GPU

```bash
cd /path/to/Kronos/Kronos
source venv/bin/activate  # or activate your conda environment

# Run with 1 GPU
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

### Multiple GPUs (Faster)

If you have multiple GPUs (e.g., 4 GPUs on p3.8xlarge):

```bash
# Run with 4 GPUs
torchrun --standalone --nproc_per_node=4 finetune/train_tokenizer.py

# Or specify specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 finetune/train_tokenizer.py
```

### Monitor Training

The script will:
- Download pretrained models from Hugging Face (first time only)
- Show training progress with loss values
- Save checkpoints to `./outputs/models/finetune_tokenizer_us_market/checkpoints/`

## Step 4: Run Predictor Finetuning

After tokenizer finetuning completes, finetune the predictor:

### Single GPU

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py
```

### Multiple GPUs

```bash
torchrun --standalone --nproc_per_node=4 finetune/train_predictor.py
```

## Step 5: Run Backtesting (Optional)

After both finetuning steps complete, evaluate the model:

```bash
# Use GPU 0 for inference
python finetune/qlib_test.py --device cuda:0

# Or use CPU (slower)
python finetune/qlib_test.py --device cpu
```

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in `finetune/config.py`:
```python
self.batch_size = 25  # Reduce from 50 to 25 or lower
```

### Issue: "NCCL backend not available"

**Solution**: Ensure you're using `torchrun` (not `python` directly):
```bash
# Correct
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py

# Incorrect
python finetune/train_tokenizer.py
```

### Issue: "CUDA not available"

**Solution**: 
1. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Issue: Models not downloading

**Solution**: Ensure internet connectivity and Hugging Face access:
```bash
# Test Hugging Face access
python -c "from huggingface_hub import hf_hub_download; print('HF access OK')"
```

## Performance Tips

1. **Use multiple GPUs** if available - significantly faster training
2. **Increase batch size** if you have GPU memory headroom
3. **Use mixed precision** (already enabled in the code with `half=True`)
4. **Monitor GPU usage**: `watch -n 1 nvidia-smi`

## Expected Training Times

On `g5.2xlarge` (1x A10G GPU):
- Tokenizer finetuning: ~2-4 hours
- Predictor finetuning: ~4-8 hours
- Total: ~6-12 hours

On `p3.8xlarge` (4x V100 GPUs):
- Tokenizer finetuning: ~1-2 hours
- Predictor finetuning: ~2-4 hours
- Total: ~3-6 hours

## Output Locations

After training completes, you'll find:

- **Tokenizer checkpoints**: `./outputs/models/finetune_tokenizer_us_market/checkpoints/best_model/`
- **Predictor checkpoints**: `./outputs/models/finetune_predictor_us_market/checkpoints/best_model/`
- **Training summary**: `./outputs/models/*/summary.json`
- **Backtest results**: `./outputs/backtest_results/`

## Quick Start Script

Create a script `run_training.sh`:

```bash
#!/bin/bash
set -e

echo "=== Step 1: Verify GPU ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Step 2: Finetune Tokenizer ==="
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py

echo "=== Step 3: Finetune Predictor ==="
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py

echo "=== Step 4: Run Backtesting ==="
python finetune/qlib_test.py --device cuda:0

echo "=== Training Complete! ==="
```

Make it executable and run:
```bash
chmod +x run_training.sh
./run_training.sh
```


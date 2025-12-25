# Fix: NameError: name 'safetensors' is not defined

## The Problem

The error occurs because the `safetensors` package is either:
1. Not installed
2. Version incompatible with `huggingface_hub`

## Solution: Install/Upgrade safetensors

On your AWS instance, run:

```bash
source venv/bin/activate

# Install or upgrade safetensors
pip install --upgrade safetensors

# Also ensure huggingface_hub is up to date
pip install --upgrade huggingface_hub
```

## Verify Installation

```bash
python -c "import safetensors; print('safetensors version:', safetensors.__version__)"
```

## Then Retry Training

After installing, run the training again:

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

## Alternative: Install All Dependencies

If you want to ensure everything is installed:

```bash
source venv/bin/activate

# Install from requirements
pip install -r requirements.txt

# Install/upgrade specific packages
pip install --upgrade safetensors huggingface_hub

# Install qlib
pip install git+https://github.com/microsoft/qlib.git

# Install comet_ml (even if not used)
pip install comet_ml
```


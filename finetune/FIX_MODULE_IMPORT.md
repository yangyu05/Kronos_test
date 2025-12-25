# Fix: ModuleNotFoundError: No module named 'model'

## The Problem

The training scripts use relative paths (`../`) which can fail depending on where you run the command from.

## Solution 1: Run from the Correct Directory (Quick Fix)

Make sure you're in the **Kronos root directory** (not the finetune directory) when running:

```bash
# On your AWS instance
cd /path/to/Kronos/Kronos  # Make sure you're in the root directory

# Then run from here
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

## Solution 2: Use Absolute Paths (Already Fixed)

I've updated the scripts to use absolute paths. If you've pulled the latest code, this should work from any directory.

## Solution 3: Add Project Root to PYTHONPATH

Alternatively, you can set the PYTHONPATH environment variable:

```bash
# On your AWS instance
export PYTHONPATH=/path/to/Kronos/Kronos:$PYTHONPATH

# Then run
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```

## Verify the Fix

Test if the import works:

```bash
cd /path/to/Kronos/Kronos
python -c "import sys; sys.path.insert(0, '.'); from model.kronos import KronosTokenizer; print('Import successful!')"
```

## Recommended: Always Run from Project Root

The easiest approach is to always run commands from the Kronos root directory:

```bash
# Always start here
cd /path/to/Kronos/Kronos
source venv/bin/activate

# Then run any script
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py
python finetune/qlib_data_preprocess.py
python finetune/qlib_test.py --device cuda:0
```


# Fix: ModuleNotFoundError: No module named 'comet_ml'

## Quick Fix (Recommended)

Even though Comet ML is disabled in your config, the training scripts import it. Simply install it:

```bash
# On your AWS instance
source venv/bin/activate
pip install comet_ml
```

That's it! The script will import it but won't use it since `use_comet = False` in your config.

## Alternative: Install All Missing Dependencies

If you want to make sure everything is installed:

```bash
source venv/bin/activate

# Install comet_ml
pip install comet_ml

# Or install from requirements if it's there
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import comet_ml; print('comet_ml installed successfully')"
```

## Then Continue Training

After installing, continue with:

```bash
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```


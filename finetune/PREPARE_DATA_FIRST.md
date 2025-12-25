# Prepare Data Before Training

Before running the training scripts, you need to process the US market data first.

## Step 1: Verify Qlib Data is Downloaded

On your AWS instance, check if US market data exists:

```bash
# Check if data directory exists
ls -la ~/.qlib/qlib_data/us_data

# If it doesn't exist, download it first
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

## Step 2: Process the Data

Once Qlib data is downloaded, process it for training:

```bash
# Make sure you're in the Kronos directory
cd /path/to/Kronos/Kronos

# Activate your virtual environment
source venv/bin/activate

# Run the data preprocessing script
python finetune/qlib_data_preprocess.py
```

This will:
- Load raw data from Qlib
- Process and clean the data
- Split into train/validation/test sets
- Save to `./data/processed_datasets/`

## Step 3: Verify Processed Data

After preprocessing completes, verify the files exist:

```bash
ls -lh data/processed_datasets/*.pkl
```

You should see:
- `train_data.pkl` (largest file, ~35MB+)
- `val_data.pkl` (~164KB)
- `test_data.pkl` (~164KB)

## Step 4: Then Run Training

Only after data is processed, run the training:

```bash
# Finetune tokenizer
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py

# Finetune predictor
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py
```

## Troubleshooting

### Issue: "No data found" or "qlib data path not found"

**Solution**: Download US market data first:
```bash
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

### Issue: "ModuleNotFoundError: No module named 'qlib'"

**Solution**: Install qlib:
```bash
pip install git+https://github.com/microsoft/qlib.git
```

### Issue: Data processing is slow

**Solution**: This is normal. Processing 500+ stocks can take 5-15 minutes.

## Complete Setup Sequence

Here's the full sequence from scratch:

```bash
# 1. Install dependencies
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/microsoft/qlib.git

# 2. Download US market data (if not already done)
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

# 3. Process the data
python finetune/qlib_data_preprocess.py

# 4. Verify data exists
ls -lh data/processed_datasets/*.pkl

# 5. Now run training
torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py
```


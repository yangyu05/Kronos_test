# Dataset Usage in retrain_model.sh

## Overview

The `retrain_model.sh` script uses **preprocessed pickle files** created by `qlib_data_preprocess.py`. These files contain data split into train/validation/test sets based on date ranges defined in `config.py`.

## Data Flow

### Step 1: Data Preprocessing (`qlib_data_preprocess.py`)

**Source:** Qlib data from `~/.qlib/qlib_data/us_data`

**Date Range Loaded:** 
- From: `2011-01-01` (config.dataset_begin_time)
- To: `2025-12-29` (config.dataset_end_time)
- Instrument: `sp500` (S&P 500 stocks)

**Output Files Created:**
- `finetune/data/processed_datasets/train_data.pkl`
- `finetune/data/processed_datasets/val_data.pkl`
- `finetune/data/processed_datasets/test_data.pkl`

**Date Splits:**
- **Train**: 2011-01-01 to 2022-12-31 (historical data)
- **Val**: 2022-09-01 to 2024-05-31 (recent but not latest)
- **Test**: 2024-04-01 to 2025-12-29 (most recent, includes all 2025 data)

### Step 2: Tokenizer Training (`train_tokenizer.py`)

**Uses:**
- `QlibDataset('train')` → loads `finetune/data/processed_datasets/train_data.pkl`
- `QlibDataset('val')` → loads `finetune/data/processed_datasets/val_data.pkl`

**Date Coverage:**
- Training on: **2011-01-01 to 2022-12-31**
- Validating on: **2022-09-01 to 2024-05-31**

### Step 3: Predictor Training (`train_predictor.py`)

**Uses:**
- `QlibDataset('train')` → loads `finetune/data/processed_datasets/train_data.pkl`
- `QlibDataset('val')` → loads `finetune/data/processed_datasets/val_data.pkl`

**Date Coverage:**
- Training on: **2011-01-01 to 2022-12-31**
- Validating on: **2022-09-01 to 2024-05-31**

### Step 4: Backtesting (`qlib_test.py`)

**Uses:**
- `finetune/data/processed_datasets/test_data.pkl` (unless `--reload-from-qlib` is used)

**Date Coverage:**
- Testing on: **2024-04-01 to 2025-12-29** (includes all 2025 data)

## Summary Table

| Step | Script | Dataset File | Date Range | Purpose |
|------|--------|--------------|------------|---------|
| 2 | `qlib_data_preprocess.py` | Creates pickle files | 2011-01-01 to 2025-12-29 | Data preprocessing |
| 3 | `train_tokenizer.py` | `train_data.pkl` | 2011-01-01 to 2022-12-31 | Tokenizer training |
| 3 | `train_tokenizer.py` | `val_data.pkl` | 2022-09-01 to 2024-05-31 | Tokenizer validation |
| 4 | `train_predictor.py` | `train_data.pkl` | 2011-01-01 to 2022-12-31 | Predictor training |
| 4 | `train_predictor.py` | `val_data.pkl` | 2022-09-01 to 2024-05-31 | Predictor validation |
| 5 | `qlib_test.py` | `test_data.pkl` | 2024-04-01 to 2025-12-29 | Model evaluation |

## Important Notes

1. **2025 Data is Included**: The test dataset includes data from 2024-04-01 to 2025-12-29, so all 2025 data will be used for evaluation.

2. **Training Uses Historical Data**: The model is trained on 2011-2022 data and validated on 2022-2024 data, then tested on 2024-2025 data. This ensures the model is evaluated on truly unseen future data.

3. **Data Source**: All data comes from Qlib's S&P 500 instrument list, loaded from `~/.qlib/qlib_data/us_data`.

4. **Reprocessing Required**: If you update Qlib data, you must rerun `qlib_data_preprocess.py` to regenerate the pickle files with new data.

## Verification

To verify which datasets are being used, check:

```bash
# Check config
python3 -c "from finetune.config import Config; c=Config(); print(f'Train: {c.train_time_range}'); print(f'Val: {c.val_time_range}'); print(f'Test: {c.test_time_range}')"

# Check file timestamps
ls -lh finetune/data/processed_datasets/*.pkl

# Check data ranges in pickle files
python3 -c "
import pickle
from pathlib import Path
config_path = Path('finetune/data/processed_datasets')
for split in ['train', 'val', 'test']:
    with open(config_path / f'{split}_data.pkl', 'rb') as f:
        data = pickle.load(f)
    if data:
        all_dates = []
        for symbol, df in data.items():
            if len(df) > 0:
                all_dates.extend(df.index.tolist())
        if all_dates:
            import pandas as pd
            dates = pd.to_datetime(all_dates)
            print(f'{split}: {dates.min().date()} to {dates.max().date()} ({len(set(all_dates))} unique dates)')
"
```


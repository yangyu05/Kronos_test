# Configuration Summary for US Market Experiment

This document summarizes the configuration for the Kronos finetuning experiment on US market data, based on the [official Kronos repository](https://github.com/shiyu-coder/Kronos).

## ✅ Configuration Status

All required paths and settings have been configured according to the Kronos repository requirements.

## 📋 Key Configuration Settings

### 1. Market Region & Data
- **Market Region**: `us` (US market)
- **Qlib Data Path**: `~/.qlib/qlib_data/us_data`
- **Instrument**: `sp500` (S&P 500 stocks)
  - Other options: `nasdaq100`, `dow30`, `all`
- **Benchmark**: `SPY` (S&P 500 ETF)

### 2. Pretrained Models (Hugging Face)
- **Tokenizer**: `NeoQuasar/Kronos-Tokenizer-base`
- **Predictor Model**: `NeoQuasar/Kronos-small`
  - Available alternatives: `NeoQuasar/Kronos-mini`, `NeoQuasar/Kronos-base`

### 3. Data Paths
- **Dataset Path**: `./data/processed_datasets`
  - Will contain: `train_data.pkl`, `val_data.pkl`, `test_data.pkl`
- **Model Save Path**: `./outputs/models`
- **Backtest Results Path**: `./outputs/backtest_results`

### 4. Time Ranges
- **Data Loading**: 2011-01-01 to 2025-06-05
- **Training**: 2011-01-01 to 2022-12-31
- **Validation**: 2022-09-01 to 2024-06-30
- **Test**: 2024-04-01 to 2025-06-05
- **Backtest**: 2024-07-01 to 2025-06-05

### 5. Model Parameters
- **Lookback Window**: 90 time steps
- **Predict Window**: 10 time steps
- **Max Context**: 512 (matches Kronos-small architecture)
- **Features**: `['open', 'high', 'low', 'close', 'vol', 'amt']`

### 6. Training Hyperparameters
- **Epochs**: 30
- **Batch Size**: 50 per GPU
- **Tokenizer Learning Rate**: 2e-4
- **Predictor Learning Rate**: 4e-5
- **Training Samples per Epoch**: 100,000 (2000 * batch_size)
- **Validation Samples**: 20,000 (400 * batch_size)

### 7. Experiment Logging
- **Comet ML**: Disabled by default (`use_comet = False`)
  - To enable: Set `use_comet = True` and configure API key via environment variable
  - `export COMET_API_KEY=your_key`
  - `export COMET_WORKSPACE=your_workspace`

### 8. Backtesting Parameters
- **Symbols to Hold**: 50
- **Symbols to Drop**: 5
- **Minimum Hold Period**: 5 days
- **Inference Temperature**: 0.6
- **Top-p Sampling**: 0.9

## 🚀 Next Steps

### Step 1: Verify Data is Downloaded
```bash
# Check if US market data exists
ls ~/.qlib/qlib_data/us_data
```

### Step 2: Process the Data
```bash
cd finetune
python qlib_data_preprocess.py
```

This will create the processed datasets in `./data/processed_datasets/`.

### Step 3: Finetune the Tokenizer
```bash
# Single GPU
python finetune/train_tokenizer.py

# Multi-GPU (replace NUM_GPUS with your GPU count)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

### Step 4: Finetune the Predictor
```bash
# Single GPU
python finetune/train_predictor.py

# Multi-GPU (replace NUM_GPUS with your GPU count)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

### Step 5: Run Backtesting
```bash
python finetune/qlib_test.py --device cuda:0
```

## 📝 Notes

1. **Model Download**: The pretrained models will be automatically downloaded from Hugging Face Hub on first use. Ensure you have internet connectivity.

2. **Directory Creation**: All output directories (`./data/processed_datasets`, `./outputs/models`, `./outputs/backtest_results`) will be created automatically if they don't exist.

3. **GPU Requirements**: 
   - Kronos-small requires significant GPU memory
   - Multi-GPU training is recommended for faster training
   - Adjust `batch_size` if you encounter out-of-memory errors

4. **Data Requirements**: 
   - Ensure US market data is downloaded and accessible
   - The preprocessing script will verify data availability

5. **Customization**: 
   - Adjust `instrument` to use different stock universes (nasdaq100, dow30, all)
   - Modify time ranges to match your data availability
   - Adjust hyperparameters based on your specific needs

## 🔗 References

- [Kronos GitHub Repository](https://github.com/shiyu-coder/Kronos)
- [Kronos Hugging Face Models](https://huggingface.co/NeoQuasar)
- [Qlib Documentation](https://qlib.readthedocs.io/)


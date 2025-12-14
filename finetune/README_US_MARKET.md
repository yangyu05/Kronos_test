# US Market Data Preparation Guide

This guide explains how to prepare US market data for the Kronos model using Qlib.

## Overview

The Kronos model can be trained on US market data using Microsoft's Qlib platform. This guide walks you through the process of downloading and preparing US market data.

## Prerequisites

- Python 3.8+ with qlib installed
- Internet connection for downloading market data
- Sufficient disk space (~5-10 GB for full US market data)

## Step 1: Download US Market Data

Use the provided script to download US market data from Qlib:

```bash
cd finetune
python prepare_us_data.py
```

This script will:
- Download US market data using Qlib's built-in data downloader
- Store the data in `~/.qlib/qlib_data/us_data`
- Verify that the data was downloaded correctly

**Note**: The download process may take 30-60 minutes depending on your internet connection, as it downloads historical data for all US stocks.

### Alternative: Manual Download

If you prefer to download data manually, you can use Qlib's CLI directly:

```bash
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

## Step 2: Configure for US Market

The configuration file (`config.py`) has been updated to support US market. By default, it's set to use US market:

```python
self.market_region = 'us'  # Set to 'cn' for Chinese market
self.qlib_data_path = "~/.qlib/qlib_data/us_data"
self.instrument = 'sp500'  # Options: 'sp500', 'nasdaq100', 'dow30', 'all'
```

### Available US Market Instruments

- `sp500`: S&P 500 stocks (default)
- `nasdaq100`: NASDAQ 100 stocks
- `dow30`: Dow Jones Industrial Average (30 stocks)
- `all`: All available US stocks (largest dataset)

### Adjust Time Ranges

Update the time ranges in `config.py` to match your needs:

```python
self.dataset_begin_time = "2011-01-01"  # Start date for data loading
self.dataset_end_time = '2025-06-05'     # End date for data loading

self.train_time_range = ["2011-01-01", "2022-12-31"]
self.val_time_range = ["2022-09-01", "2024-06-30"]
self.test_time_range = ["2024-04-01", "2025-06-05"]
```

## Step 3: Process the Data

After downloading the data, process it for training:

```bash
python qlib_data_preprocess.py
```

This script will:
- Load raw market data from Qlib
- Process and clean the data
- Split into train/validation/test sets
- Save processed datasets as pickle files

The processed datasets will be saved to the directory specified in `config.py`:
```python
self.dataset_path = "./data/processed_datasets"
```

## Step 4: Verify Data

You can verify that the data was processed correctly by checking:

1. **Data directory exists**: `~/.qlib/qlib_data/us_data`
2. **Processed datasets exist**: `./data/processed_datasets/train_data.pkl`, `val_data.pkl`, `test_data.pkl`

## Troubleshooting

### Issue: Data download fails

**Solution**: 
- Check your internet connection
- Ensure you have sufficient disk space
- Try downloading again (the script will skip if data already exists)

### Issue: "No data found" error

**Solution**:
- Verify that data was downloaded: `ls ~/.qlib/qlib_data/us_data`
- Check that `qlib_data_path` in `config.py` matches the download location
- Ensure `market_region` is set to `'us'` in `config.py`

### Issue: Instrument not found

**Solution**:
- Verify the instrument name (case-sensitive): `'sp500'`, `'nasdaq100'`, `'dow30'`, or `'all'`
- Check Qlib documentation for available instruments: https://qlib.readthedocs.io/

### Issue: Out of memory during processing

**Solution**:
- Use a smaller instrument (e.g., `'dow30'` instead of `'all'`)
- Reduce the time range
- Process data in batches (modify the preprocessing script)

## Additional Resources

- [Qlib Documentation](https://qlib.readthedocs.io/)
- [Qlib GitHub Repository](https://github.com/microsoft/qlib)
- [Qlib US Market Data Guide](https://qlib.readthedocs.io/en/latest/component/data.html#us-market-data)

## Switching Between Markets

To switch between US and Chinese markets, simply change the `market_region` in `config.py`:

```python
# For US market
self.market_region = 'us'
self.qlib_data_path = "~/.qlib/qlib_data/us_data"
self.instrument = 'sp500'

# For Chinese market
self.market_region = 'cn'
self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
self.instrument = 'csi300'
```

The data preprocessor will automatically use the correct region based on this setting.


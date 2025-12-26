# How to Add ETFs to Qlib

This guide explains multiple methods to add ETFs (Exchange-Traded Funds) to your Qlib dataset for use with Kronos.

## Method 1: Check if ETFs are Already Available

Qlib's US market data download may already include ETFs. First, check what's available:

```bash
cd finetune
python check_etf_availability.py
```

This script will:
- List all available instruments in your Qlib data
- Check for common ETFs like SPY, QQQ, DIA, etc.
- Show you the first 50 available instruments

If your desired ETFs are already available, you can proceed to Method 2.

## Method 2: Use Custom Symbol List (Recommended)

If ETFs are available in your Qlib data, you can use them directly by specifying a custom symbol list in `config.py`:

### Step 1: Edit `finetune/config.py`

Add your ETF symbols to the `custom_symbols` list:

```python
# In config.py, find the US market section and modify:
if self.market_region == 'us':
    self.qlib_data_path = "~/.qlib/qlib_data/us_data"
    self.instrument = 'sp500'  # This will be ignored if custom_symbols is set
    # Add your ETFs and stocks here
    self.custom_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'AAPL', 'MSFT', 'GOOGL']
```

### Step 2: Re-run Data Preprocessing

After modifying the config, reprocess your data:

```bash
cd finetune
python qlib_data_preprocess.py
```

The script will now load data for your custom symbol list (including ETFs) instead of the predefined instrument.

### Step 3: Run Backtesting with ETFs

You can now use ETFs in your backtesting:

```bash
python finetune/qlib_test.py --device cuda:0 --symbols SPY QQQ DIA
```

## Method 3: Add ETFs to Qlib Data Directory (Advanced)

If your ETFs are not in the default Qlib dataset, you can manually add them:

### Step 1: Download ETF Data

You'll need to download historical OHLCV data for your ETFs. You can use:
- Yahoo Finance API
- Alpha Vantage
- Other financial data providers

### Step 2: Format Data for Qlib

Qlib expects data in a specific format. The data structure should be:
```
~/.qlib/qlib_data/us_data/
├── calendars/
│   └── day.txt
├── instruments/
│   └── all.txt
└── features/
    ├── $close/
    │   └── day/
    │       └── {symbol}.pkl
    ├── $open/
    ├── $high/
    ├── $low/
    ├── $volume/
    └── $vwap/
```

### Step 3: Add ETF Symbols to Instruments List

Edit `~/.qlib/qlib_data/us_data/instruments/all.txt` and add your ETF symbols (one per line).

### Step 4: Create Feature Files

For each ETF symbol, create pickle files in the appropriate feature directories with the OHLCV data.

**Note**: This is a complex process. It's recommended to use Qlib's built-in data downloader or check if your ETFs are already available first.

## Method 4: Combine Stocks and ETFs

You can combine stocks from a predefined instrument with custom ETFs:

```python
# In config.py
from qlib.data import D
import qlib
from qlib.config import REG_US

# Initialize Qlib first
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)

# Get stocks from an instrument
stocks = D.instruments('sp500')

# Add your ETFs
etfs = ['SPY', 'QQQ', 'DIA']

# Combine them
self.custom_symbols = list(stocks) + etfs
```

## Verification

After adding ETFs, verify they're being loaded:

1. Check the preprocessing output - it should list your custom symbols
2. Check the processed data files - ETFs should appear in `train_data.pkl`, `val_data.pkl`, and `test_data.pkl`
3. Run a quick test:

```python
import pickle
with open('data/processed_datasets/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    print("Available symbols:", list(test_data.keys()))
    if 'SPY' in test_data:
        print("✓ SPY ETF is available!")
```

## Troubleshooting

### Issue: "Symbol not found" error

**Solution**: 
- Verify the symbol exists in Qlib data: `python check_etf_availability.py`
- Check symbol spelling (case-sensitive)
- Ensure the symbol is in the instruments list

### Issue: ETF data has missing dates

**Solution**: 
- ETFs may have different trading calendars than stocks
- The preprocessing script will filter out symbols with insufficient data
- Consider using `'all'` instrument which may include more ETFs

### Issue: Want to use ETFs only

**Solution**: 
- Set `custom_symbols` to only ETF symbols:
  ```python
  self.custom_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
  ```

## Example: Complete Setup for ETF Backtesting

```python
# In finetune/config.py

if self.market_region == 'us':
    self.qlib_data_path = "~/.qlib/qlib_data/us_data"
    self.instrument = 'sp500'  # Fallback if custom_symbols is None
    
    # Define your ETF portfolio
    self.custom_symbols = [
        # Major ETFs
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'DIA',   # Dow Jones
        'IWM',   # Russell 2000
        'VTI',   # Total Stock Market
        # Sector ETFs
        'XLF',   # Financials
        'XLE',   # Energy
        'XLK',   # Technology
        # Add more as needed
    ]
```

Then run:
```bash
cd finetune
python qlib_data_preprocess.py
python qlib_test.py --device cuda:0 --symbols SPY QQQ DIA
```

## Additional Resources

- [Qlib Documentation](https://qlib.readthedocs.io/)
- [Qlib GitHub Repository](https://github.com/microsoft/qlib)
- [Qlib Data Format Guide](https://qlib.readthedocs.io/en/latest/component/data.html)


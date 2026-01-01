# How to Run `prediction_example.py`

## Step-by-Step Instructions

### 1. Activate Virtual Environment

```bash
cd /home/ubuntu/Kronos_test
source venv/bin/activate
```

### 2. Verify Dependencies

The script requires:
- ✅ PyTorch (with CUDA support) - Already installed
- ✅ Model classes (Kronos, KronosTokenizer, KronosPredictor) - Available
- ✅ Data file: `examples/data/XSHG_5min_600977.csv` - Exists

### 3. Run the Script

**Option A: Run as-is (will try to show plot)**
```bash
cd examples
python3 prediction_example.py
```

**Option B: Run with non-interactive backend (save plot to file)**
```bash
cd examples
python3 -c "
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
exec(open('prediction_example.py').read().replace('plt.show()', 'plt.savefig(\"prediction_result.png\")'))
"
```

### 4. Expected Output

The script will:
1. Download model and tokenizer from Hugging Face (first time only)
2. Load the data file
3. Make predictions for 120 future time steps
4. Display or save a plot comparing ground truth vs predictions

### 5. Troubleshooting

**Issue: "No module named 'torch'"**
- Solution: Make sure you activated the virtual environment: `source venv/bin/activate`

**Issue: "Cannot display plot" (headless server)**
- Solution: Modify the script to save instead of show:
  ```python
  # Change line 38 from:
  plt.show()
  # To:
  plt.savefig('prediction_result.png')
  print("Plot saved to prediction_result.png")
  ```

**Issue: "CUDA out of memory"**
- Solution: Change device to CPU in line 46:
  ```python
  predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
  ```

**Issue: "FileNotFoundError: ./data/XSHG_5min_600977.csv"**
- Solution: Make sure you're running from the `examples/` directory, or update the path in the script

### 6. Customizing the Script

You can modify these parameters:
- `lookback = 400` - Number of historical data points to use
- `pred_len = 120` - Number of future steps to predict
- `device = "cuda:0"` - Use "cpu" if no GPU available
- `T = 1.0` - Temperature for sampling (higher = more randomness)
- `top_p = 0.9` - Nucleus sampling threshold
- `sample_count = 1` - Number of forecast paths to average

## Quick Start Command

```bash
cd /home/ubuntu/Kronos_test && \
source venv/bin/activate && \
cd examples && \
python3 prediction_example.py
```


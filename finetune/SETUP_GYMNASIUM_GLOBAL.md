# Setup Gymnasium as Global Gym Replacement

## The Problem

Qlib imports `gym` internally, which doesn't support NumPy 2.0. We need to make `gymnasium` available as `gym` before qlib is imported.

## Solution: Global Alias Setup

I've updated the scripts to automatically set up gymnasium as gym replacement. The fix is already in:
- `qlib_test.py` - Updated
- `qlib_data_preprocess.py` - Updated

## Manual Setup (If Needed)

If you need to set this up in other scripts, add this at the very top (before any qlib imports):

```python
import sys
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
    print("✓ Using gymnasium as gym replacement")
except ImportError:
    print("⚠ gymnasium not found, falling back to gym")

# Now safe to import qlib
import qlib
```

## Or Use the Setup Module

You can also use the provided setup module:

```python
# At the very top of your script
import setup_gymnasium_global  # Must be first
import qlib  # Now safe
```

## Verify It Works

Test that gymnasium is being used:

```bash
python -c "
import sys
import gymnasium
sys.modules['gym'] = gymnasium
import qlib
print('✓ Success: qlib imported with gymnasium')
"
```

## Installation

Make sure gymnasium is installed:

```bash
source venv/bin/activate
pip install gymnasium
```

## Why This Works

Python's `sys.modules` is a dictionary that caches imported modules. By setting `sys.modules['gym'] = gymnasium` before qlib tries to import gym, when qlib does `import gym`, Python will return the gymnasium module instead.

This is a common pattern for creating compatibility layers between packages.


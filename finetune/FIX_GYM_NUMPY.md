# Fix: Gym NumPy 2.0 Compatibility Error

## The Problem

Gym (OpenAI Gym) is a dependency of qlib but doesn't support NumPy 2.0. You'll see errors like:
```
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
```

## Solution: Downgrade NumPy to 1.x

The easiest fix is to downgrade NumPy to version 1.x (which Gym supports):

```bash
# On your AWS instance
source venv/bin/activate

# Downgrade NumPy to latest 1.x version
pip install "numpy<2.0"

# Verify version
python -c "import numpy; print('NumPy version:', numpy.__version__)"
# Should show: NumPy version: 1.26.x or similar (not 2.x)
```

## Alternative: Pin Specific NumPy Version

If you want to be more specific:

```bash
pip install "numpy>=1.21.0,<2.0"
```

## Complete Fix with All Dependencies

To ensure all dependencies are compatible:

```bash
source venv/bin/activate

# Downgrade NumPy
pip install "numpy<2.0"

# Reinstall qlib to ensure compatibility
pip install --force-reinstall git+https://github.com/microsoft/qlib.git

# Verify
python -c "import numpy; import qlib; print('✓ NumPy:', numpy.__version__); print('✓ Qlib imported successfully')"
```

## Why This Happens

- **NumPy 2.0** was released in 2024 with breaking changes
- **Gym** (OpenAI Gym) hasn't been updated since 2022
- **Qlib** depends on Gym for some functionality
- The combination causes compatibility issues

## Verify Fix

After downgrading NumPy, test the backtesting script:

```bash
python finetune/qlib_test.py --device cuda:0
```

## If You Need NumPy 2.0 for Other Projects

If you need NumPy 2.0 for other projects, you can:

1. **Use separate virtual environments** for different projects
2. **Use conda environments** with different NumPy versions
3. **Wait for qlib/gym updates** (may take time)

For now, using NumPy 1.x is the recommended solution.

## Check Current NumPy Version

```bash
python -c "import numpy; print(numpy.__version__)"
```

If it shows `2.x`, you need to downgrade.


# Install Gymnasium to Replace Gym

Gymnasium is the maintained fork of OpenAI Gym and supports NumPy 2.0. It's designed as a drop-in replacement.

## Step 1: Uninstall Gym and Install Gymnasium

On your AWS instance:

```bash
source venv/bin/activate

# Uninstall the old gym
pip uninstall -y gym

# Install gymnasium (maintained fork, NumPy 2.0 compatible)
pip install gymnasium

# Verify installation
python -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)"
```

## Step 2: Create Compatibility Alias (If Needed)

Some code might still import `gym`. You can create a compatibility layer:

```bash
# Create a simple compatibility script
cat > /tmp/gym_compat.py << 'EOF'
# Compatibility layer: gym -> gymnasium
import sys
import gymnasium as gym
sys.modules['gym'] = gym
EOF

# Or add to your Python path
export PYTHONPATH=/tmp:$PYTHONPATH
```

## Step 3: Test Qlib with Gymnasium

```bash
python -c "
import gymnasium
import qlib
print('✓ Gymnasium imported')
print('✓ Qlib imported')
print('Testing qlib initialization...')
import qlib
from qlib.config import REG_US
qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_US)
print('✓ Qlib initialized successfully with Gymnasium')
"
```

## Alternative: Use Gymnasium with Gym Alias

If qlib specifically requires `gym` module name, you can install both:

```bash
source venv/bin/activate

# Install gymnasium
pip install gymnasium

# Install gymnasium[classic] which provides gym compatibility
pip install "gymnasium[classic]"

# Or create a symlink/compatibility import
python -c "
import sys
import gymnasium
sys.modules['gym'] = gymnasium
"
```

## Step 4: Update Your Environment

Add this to your training/backtesting scripts or create a wrapper:

```python
# Add at the top of your scripts or in a setup file
import sys
try:
    import gymnasium as gym
    sys.modules['gym'] = gym
except ImportError:
    import gym
```

## Verify It Works

After installation, test the backtesting:

```bash
python finetune/qlib_test.py --device cuda:0
```

## If Issues Persist

If qlib still has issues, you might need to:

1. **Keep both gym and gymnasium** (gymnasium can coexist)
2. **Use NumPy < 2.0** (the original solution)
3. **Check qlib version** - newer versions might support gymnasium

## Recommended Approach

The safest approach is to install gymnasium and create a compatibility alias:

```bash
source venv/bin/activate

# Install gymnasium
pip install gymnasium

# Test if it works
python -c "
import sys
import gymnasium
sys.modules['gym'] = gymnasium
import qlib
print('✓ Success: Gymnasium works with qlib')
"
```


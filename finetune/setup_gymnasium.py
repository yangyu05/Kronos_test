"""
Setup script to use Gymnasium instead of Gym.
Run this before importing qlib or running training/backtesting scripts.
"""
import sys

try:
    import gymnasium
    # Make gymnasium available as 'gym' for backward compatibility
    sys.modules['gym'] = gymnasium
    print("✓ Gymnasium loaded and aliased as 'gym'")
except ImportError:
    print("⚠ Gymnasium not found, falling back to gym")
    try:
        import gym
        print("✓ Using gym (may have NumPy 2.0 issues)")
    except ImportError:
        print("✗ Neither gymnasium nor gym found!")
        sys.exit(1)

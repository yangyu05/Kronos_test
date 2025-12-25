"""
Global setup to use gymnasium as gym replacement.
Import this module at the very beginning of any script that uses qlib.

Usage:
    import setup_gymnasium_global  # Must be first import
    import qlib  # Now safe to import
"""
import sys

try:
    import gymnasium
    sys.modules['gym'] = gymnasium
    print("✓ Gymnasium loaded and aliased as 'gym' globally")
except ImportError:
    print("⚠ Warning: gymnasium not found. Install with: pip install gymnasium")
    try:
        import gym
        print("⚠ Falling back to gym (may have NumPy 2.0 compatibility issues)")
    except ImportError:
        print("✗ Error: Neither gymnasium nor gym found!")
        raise


#!/usr/bin/env python3
"""
Script to check if ETFs are available in Qlib data.
"""
import sys
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    pass

import qlib
from qlib.config import REG_US
from qlib.data import D

# Initialize Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)

# Check available instruments
print("Checking available instruments...")
all_instruments = D.instruments(market='all')
print(f"Total instruments available: {len(all_instruments)}")

# Check for common ETFs
etf_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'TLT']
print("\nChecking for common ETFs:")
for etf in etf_symbols:
    if etf in all_instruments:
        print(f"  ✓ {etf} is available")
    else:
        print(f"  ✗ {etf} is NOT available")

# Show first 50 instruments as examples
print(f"\nFirst 50 available instruments:")
for i, inst in enumerate(list(all_instruments)[:50]):
    print(f"  {inst}", end="  ")
    if (i + 1) % 10 == 0:
        print()


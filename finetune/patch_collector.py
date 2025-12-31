#!/usr/bin/env python3
"""
Patch Qlib collector to use custom symbols instead of fetching from external APIs.
"""
import sys
import os
from pathlib import Path

# Add qlib_repo to path
qlib_repo = Path.home() / 'qlib_repo'
if not qlib_repo.exists():
    print(f"❌ Qlib repo not found at {qlib_repo}")
    print("Please clone it: git clone https://github.com/microsoft/qlib.git ~/qlib_repo")
    sys.exit(1)

# Patch the collector file
collector_file = qlib_repo / 'scripts' / 'data_collector' / 'yahoo' / 'collector.py'

if not collector_file.exists():
    print(f"❌ Collector file not found: {collector_file}")
    sys.exit(1)

# Read the file
with open(collector_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'CUSTOM_SYMBOLS_PATCH' in content:
    print("✓ Collector already patched")
else:
    # Create backup
    backup_file = collector_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✓ Created backup: {backup_file}")
    
    # Patch the get_instrument_list method
    old_method = '''    def get_instrument_list(self):
        logger.info("get US stock symbols......")
        symbols = get_us_stock_symbols() + [
            "^GSPC",
            "^NDX",
            "^DJI",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols'''
    
    new_method = '''    def get_instrument_list(self):
        # CUSTOM_SYMBOLS_PATCH: Allow custom symbols via environment variable
        import os
        custom_symbols = os.environ.get('QLIB_CUSTOM_SYMBOLS')
        if custom_symbols:
            symbols = custom_symbols.split(',') + [
                "^GSPC",
                "^NDX",
                "^DJI",
            ]
            logger.info(f"Using custom symbols: {len(symbols)} symbols")
            return symbols
        
        logger.info("get US stock symbols......")
        try:
            symbols = get_us_stock_symbols() + [
                "^GSPC",
                "^NDX",
                "^DJI",
            ]
            logger.info(f"get {len(symbols)} symbols.")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            logger.error("Set QLIB_CUSTOM_SYMBOLS environment variable with comma-separated symbols")
            raise'''
    
    if old_method in content:
        content = content.replace(old_method, new_method)
        with open(collector_file, 'w') as f:
            f.write(content)
        print("✓ Patched collector to support custom symbols")
    else:
        print("⚠️  Could not find method to patch. File may have changed.")


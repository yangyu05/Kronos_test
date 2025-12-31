#!/usr/bin/env python3
"""
Simple script to update Qlib data using existing instruments or custom symbol list.
This bypasses the collector's symbol fetching which can fail.
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add qlib_repo to path
sys.path.insert(0, str(Path.home() / 'qlib_repo'))

try:
    from scripts.data_collector.yahoo.collector import YahooCollectorUS
    from scripts.data_collector.utils import get_us_stock_symbols
except ImportError as e:
    print(f"❌ Error importing Qlib collector: {e}")
    print("Make sure you've cloned the Qlib repo to ~/qlib_repo")
    sys.exit(1)

import qlib
from qlib.config import REG_US
from qlib.data import D


class CustomYahooCollectorUS(YahooCollectorUS):
    """Custom collector that uses existing instruments or provided symbol list."""
    
    def __init__(self, custom_symbols=None, use_existing_instruments=True, *args, **kwargs):
        self.custom_symbols = custom_symbols
        self.use_existing_instruments = use_existing_instruments
        super().__init__(*args, **kwargs)
    
    def get_instrument_list(self):
        """Override to use custom symbols or existing instruments."""
        if self.custom_symbols:
            logger.info(f"Using provided symbol list: {len(self.custom_symbols)} symbols")
            return self.custom_symbols + ["^GSPC", "^NDX", "^DJI"]
        
        if self.use_existing_instruments:
            try:
                # Try to get existing instruments from Qlib
                qlib_data_path = os.path.expanduser("~/.qlib/qlib_data/us_data")
                qlib.init(provider_uri=qlib_data_path, region=REG_US)
                instruments = D.instruments('all')
                
                # Filter out non-stock symbols
                stock_symbols = [s for s in instruments if not s.startswith('^') and s not in ['market', 'filter_pipe']]
                
                if stock_symbols:
                    logger.info(f"Using existing Qlib instruments: {len(stock_symbols)} symbols")
                    return stock_symbols + ["^GSPC", "^NDX", "^DJI"]
            except Exception as e:
                logger.warning(f"Could not get existing instruments: {e}")
        
        # Fallback: try to get symbols from external APIs (may fail)
        logger.info("Attempting to fetch symbols from external APIs...")
        try:
            symbols = get_us_stock_symbols() + ["^GSPC", "^NDX", "^DJI"]
            logger.info(f"Fetched {len(symbols)} symbols from external APIs")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            logger.error("Please provide symbols using --symbols argument")
            raise ValueError("Could not get instrument list. Use --symbols to provide symbols.")


def main():
    parser = argparse.ArgumentParser(
        description="Update Qlib data using custom symbols or existing instruments"
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to update (e.g., AAPL MSFT GOOGL). If not provided, uses existing Qlib instruments.'
    )
    parser.add_argument(
        '--qlib-path',
        default='~/.qlib/qlib_data/us_data',
        help='Path to Qlib data directory'
    )
    parser.add_argument(
        '--start-date',
        default='2020-11-11',
        help='Start date for data collection'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date for data collection (default: today)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching symbols from external APIs, only use provided or existing symbols'
    )
    
    args = parser.parse_args()
    
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("Qlib Data Updater (Custom Symbol List)")
    print("=" * 60)
    print(f"Qlib path: {args.qlib_path}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    if args.symbols:
        print(f"Symbols: {args.symbols}")
    else:
        print("Symbols: Using existing Qlib instruments")
    print("=" * 60)
    
    # Create collector with custom symbol list
    collector = CustomYahooCollectorUS(
        custom_symbols=args.symbols,
        use_existing_instruments=not args.skip_fetch,
        save_dir=os.path.join(os.path.expanduser('~'), 'qlib_temp_download'),
        start=args.start_date,
        end=args.end_date,
    )
    
    # Run update
    try:
        collector.update_data_to_bin(
            qlib_data_1d_dir=os.path.expanduser(args.qlib_path),
            trading_date=args.start_date,
            end_date=args.end_date,
        )
        print("\n✅ Data update completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Import logger
    from scripts.data_collector.utils import logger
    main()


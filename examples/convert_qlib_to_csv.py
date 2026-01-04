#!/usr/bin/env python3
"""
Convert Qlib data to CSV format for prediction_example.py

Usage:
    python convert_qlib_to_csv.py --symbol AAPL --output data/AAPL_1d.csv
    python convert_qlib_to_csv.py --symbol AAPL --start-date 2020-01-01 --end-date 2025-12-31
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import qlib
from qlib.config import REG_US, REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from finetune.config import Config


def convert_qlib_to_csv(
    symbol: str,
    output_path: str = None,
    start_date: str = None,
    end_date: str = None,
    qlib_data_path: str = None,
    region: str = None,
    freq: str = None
):
    """
    Convert Qlib data for a symbol to CSV format compatible with prediction_example.py
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        output_path: Output CSV file path (default: examples/data/{symbol}_1d.csv)
        start_date: Start date (YYYY-MM-DD). If None, uses available data from Qlib
        end_date: End date (YYYY-MM-DD). If None, uses latest available date
        qlib_data_path: Path to Qlib data directory
        region: Market region ('us' or 'cn')
        freq: Data frequency ('day', '5min', '15min', etc.). If None, tries to detect from config
    """
    # Use config defaults if not provided
    config = Config()
    if qlib_data_path is None:
        qlib_data_path = config.qlib_data_path
    if region is None:
        region = config.market_region
    if freq is None:
        # Try to detect frequency from config
        freq = getattr(config, 'data_freq', 'day')
    
    # Initialize Qlib
    qlib_region = REG_US if region == 'us' else REG_CN
    print(f"Initializing Qlib with region: {region} ({qlib_region})")
    print(f"Data path: {qlib_data_path}")
    print(f"Frequency: {freq}")
    qlib.init(provider_uri=qlib_data_path, region=qlib_region)
    
    # Get calendar to determine available dates (use specified frequency)
    if freq == 'day':
        cal = D.calendar()
    else:
        cal = D.calendar(freq=freq)
    if len(cal) == 0:
        raise ValueError(f"No calendar data found in Qlib for frequency '{freq}'!")
    
    # For intraday data, show datetime range; for daily, show date range
    if freq == 'day':
        print(f"\nCalendar range: {cal[0].date()} to {cal[-1].date()}")
    else:
        print(f"\nCalendar range: {cal[0]} to {cal[-1]}")
    
    # Determine time range
    if start_date is None:
        start_date = cal[0].strftime('%Y-%m-%d')
        print(f"Using earliest available date: {start_date}")
    else:
        start_ts = pd.Timestamp(start_date)
        if start_ts < cal[0]:
            start_date = cal[0].strftime('%Y-%m-%d')
            print(f"⚠️  Requested start_date is before available data. Using: {start_date}")
    
    if end_date is None:
        end_date = cal[-1].strftime('%Y-%m-%d')
        print(f"Using latest available date: {end_date}")
    else:
        end_ts = pd.Timestamp(end_date)
        if end_ts > cal[-1]:
            end_date = cal[-1].strftime('%Y-%m-%d')
            print(f"⚠️  Requested end_date exceeds available data. Using: {end_date}")
    
    # Load data fields (matching prediction script requirements)
    data_fields_qlib = ['$open', '$high', '$low', '$close', '$volume']
    
    print(f"\nLoading data for {symbol} from {start_date} to {end_date}...")
    
    try:
        # Try loading directly using D.features (doesn't require instrument file)
        try:
            print("  Attempting to load via D.features...")
            # For intraday data, include time in start/end times if not already present
            if freq != 'day':
                # For intraday, ensure we have full datetime range
                start_time_str = f"{start_date} 00:00:00" if len(start_date) == 10 else start_date
                end_time_str = f"{end_date} 23:59:59" if len(end_date) == 10 else end_date
            else:
                start_time_str = start_date
                end_time_str = end_date
            
            data_df = D.features([symbol.upper()], data_fields_qlib, start_time=start_time_str, end_time=end_time_str, freq=freq)
            
            if data_df.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            # Handle MultiIndex: select the symbol and unstack field level
            if isinstance(data_df.index, pd.MultiIndex):
                # If MultiIndex has 'instrument' level, select the symbol
                if 'instrument' in data_df.index.names:
                    symbol_df = data_df.xs(symbol.upper(), level='instrument')
                else:
                    symbol_df = data_df
                # Unstack field level to get columns
                if 'field' in symbol_df.index.names or len(symbol_df.index.names) > 1:
                    symbol_df = symbol_df.unstack(level=-1)  # Unstack the last level (field)
                    symbol_df.columns = symbol_df.columns.droplevel(0) if isinstance(symbol_df.columns, pd.MultiIndex) else symbol_df.columns
            else:
                symbol_df = data_df
            
            # Clean column names (remove $ prefix)
            symbol_df.columns = [col.replace('$', '') if isinstance(col, str) else str(col).replace('$', '') for col in symbol_df.columns]
            
        except Exception as e1:
            print(f"  D.features failed: {e1}")
            print("  Attempting to load via QlibDataLoader...")
            # Fallback to QlibDataLoader
            data_df = QlibDataLoader(config=data_fields_qlib).load(
                symbol, start_date, end_date
            )
            
            if data_df.empty:
                raise ValueError(f"No data found for {symbol} in the specified date range")
            
            # Reshape data: unstack to get columns as features
            symbol_df = data_df[symbol].unstack(level=1)
            symbol_df.columns = [col.replace('$', '') for col in symbol_df.columns]
        
        # Rename 'vol' to 'volume' if needed
        if 'vol' in symbol_df.columns and 'volume' not in symbol_df.columns:
            symbol_df = symbol_df.rename(columns={'vol': 'volume'})
        
        # Calculate 'amount' if not present (amount = typical_price * volume)
        if 'amount' not in symbol_df.columns:
            if all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                typical_price = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4
                symbol_df = symbol_df.copy()  # Avoid SettingWithCopyWarning
                symbol_df['amount'] = typical_price * symbol_df['volume']
                print("  ✓ Calculated 'amount' column from typical_price * volume")
            else:
                # Fill with zeros if we can't calculate
                symbol_df = symbol_df.copy()  # Avoid SettingWithCopyWarning
                symbol_df['amount'] = 0.0
                print("  ⚠️  Could not calculate 'amount', filling with zeros")
        
        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        missing_cols = [col for col in required_cols if col not in symbol_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with NaN values
        initial_len = len(symbol_df)
        symbol_df = symbol_df[required_cols].dropna()
        removed = initial_len - len(symbol_df)
        if removed > 0:
            print(f"  ⚠️  Removed {removed} rows with NaN values")
        
        if len(symbol_df) == 0:
            raise ValueError("No valid data after cleaning")
        
        # Reset index to get datetime as a column
        symbol_df = symbol_df.reset_index()
        symbol_df = symbol_df.rename(columns={'datetime': 'timestamps'})
        
        # Ensure timestamps is datetime type
        symbol_df['timestamps'] = pd.to_datetime(symbol_df['timestamps'])
        
        # Reorder columns to match expected format
        column_order = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        symbol_df = symbol_df[column_order]
        
        # Sort by timestamps
        symbol_df = symbol_df.sort_values('timestamps').reset_index(drop=True)
        
        print(f"  ✓ Loaded {len(symbol_df)} data points")
        print(f"    Date range: {symbol_df['timestamps'].min().date()} to {symbol_df['timestamps'].max().date()}")
        print(f"    Columns: {', '.join(symbol_df.columns)}")
        
        # Determine output path
        if output_path is None:
            output_dir = Path(__file__).parent / "data"
            output_dir.mkdir(exist_ok=True)
            # Use frequency in filename if not daily
            if freq == 'day':
                output_path = output_dir / f"{symbol}_1d.csv"
            else:
                output_path = output_dir / f"{symbol}_{freq}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        symbol_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Show sample
        print(f"\n📊 Sample data (first 5 rows):")
        print(symbol_df.head().to_string(index=False))
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error loading data for {symbol}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qlib data to CSV format for prediction_example.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert AAPL data (uses all available dates)
  python convert_qlib_to_csv.py --symbol AAPL
  
  # Convert with specific date range
  python convert_qlib_to_csv.py --symbol AAPL --start-date 2020-01-01 --end-date 2025-12-31
  
  # Specify custom output path
  python convert_qlib_to_csv.py --symbol AAPL --output data/AAPL_daily.csv
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help="Stock symbol (e.g., AAPL, MSFT)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output CSV file path (default: examples/data/{symbol}_1d.csv)"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If not provided, uses earliest available date"
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If not provided, uses latest available date"
    )
    parser.add_argument(
        '--qlib-path',
        type=str,
        default=None,
        help="Path to Qlib data directory (default: from config)"
    )
    parser.add_argument(
        '--region',
        type=str,
        choices=['us', 'cn'],
        default=None,
        help="Market region (default: from config)"
    )
    parser.add_argument(
        '--freq',
        type=str,
        default=None,
        choices=['day', '1min', '5min', '15min', '30min', '1h'],
        help="Data frequency (default: from config, or 'day')"
    )
    
    args = parser.parse_args()
    
    try:
        convert_qlib_to_csv(
            symbol=args.symbol,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            qlib_data_path=args.qlib_path,
            region=args.region,
            freq=args.freq
        )
    except Exception as e:
        print(f"\n❌ Failed to convert data: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


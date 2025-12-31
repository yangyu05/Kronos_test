"""
Script to check the available time range in Qlib data.
Can also check data availability for specific symbols.
"""
import argparse
import os
import pandas as pd
import qlib
from qlib.config import REG_US, REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from config import Config

def check_qlib_time_range():
    """Check the available time range in Qlib data."""
    config = Config()
    
    # Initialize Qlib
    region = REG_US if config.market_region == 'us' else REG_CN
    print(f"Initializing Qlib with region: {config.market_region} ({region})")
    print(f"Data path: {config.qlib_data_path}")
    qlib.init(provider_uri=config.qlib_data_path, region=region)
    
    # Get the calendar (list of all available trading dates)
    cal = D.calendar()
    
    if len(cal) == 0:
        print("\n⚠️  WARNING: No calendar data found! Please check your Qlib data path.")
        return
    
    # Get first and last dates
    first_date = cal[0]
    last_date = cal[-1]
    
    print(f"\n{'='*60}")
    print(f"Qlib Data Time Range")
    print(f"{'='*60}")
    print(f"First available date: {first_date}")
    print(f"Last available date:  {last_date}")
    print(f"Total trading days:   {len(cal)}")
    print(f"{'='*60}")
    
    # Check if requested dates are available
    print(f"\n{'='*60}")
    print(f"Checking configured time ranges:")
    print(f"{'='*60}")
    
    dataset_start = config.dataset_begin_time
    dataset_end = config.dataset_end_time
    
    train_start, train_end = config.train_time_range
    val_start, val_end = config.val_time_range
    test_start, test_end = config.test_time_range
    
    print(f"\nDataset range (config):")
    print(f"  Requested: {dataset_start} to {dataset_end}")
    print(f"  Available: {first_date.date()} to {last_date.date()}")
    
    if first_date <= pd.Timestamp(dataset_start) <= last_date:
        print(f"  ✓ Start date {dataset_start} is available")
    else:
        print(f"  ✗ Start date {dataset_start} is NOT available")
    
    if first_date <= pd.Timestamp(dataset_end) <= last_date:
        print(f"  ✓ End date {dataset_end} is available")
    else:
        print(f"  ✗ End date {dataset_end} is NOT available")
    
    print(f"\nTrain range: {train_start} to {train_end}")
    if first_date <= pd.Timestamp(train_start) <= last_date and first_date <= pd.Timestamp(train_end) <= last_date:
        print(f"  ✓ Train range is within available data")
    else:
        print(f"  ✗ Train range may extend beyond available data")
    
    print(f"\nValidation range: {val_start} to {val_end}")
    if first_date <= pd.Timestamp(val_start) <= last_date and first_date <= pd.Timestamp(val_end) <= last_date:
        print(f"  ✓ Validation range is within available data")
    else:
        print(f"  ✗ Validation range may extend beyond available data")
    
    print(f"\nTest range: {test_start} to {test_end}")
    if first_date <= pd.Timestamp(test_start) <= last_date and first_date <= pd.Timestamp(test_end) <= last_date:
        print(f"  ✓ Test range is within available data")
    else:
        print(f"  ✗ Test range may extend beyond available data")
    
    print(f"\n{'='*60}")
    print(f"Recent dates (last 10 trading days):")
    for date in cal[-10:]:
        print(f"  {date.date()}")
    print(f"{'='*60}")


def check_symbol_data(symbol: str, qlib_data_path: str = None, region: str = None):
    """
    Check data availability for a specific symbol.
    
    Args:
        symbol: Stock symbol to check (e.g., 'AAPL')
        qlib_data_path: Path to Qlib data directory. If None, uses config default.
        region: Market region ('us' or 'cn'). If None, uses config default.
    """
    config = Config()
    
    # Use provided parameters or fall back to config
    if qlib_data_path is None:
        qlib_data_path = config.qlib_data_path
    if region is None:
        region = config.market_region
    
    # Initialize Qlib
    qlib_region = REG_US if region == 'us' else REG_CN
    print(f"\n{'='*60}")
    print(f"Checking data for symbol: {symbol}")
    print(f"{'='*60}")
    print(f"Qlib data path: {qlib_data_path}")
    print(f"Region: {region}")
    
    qlib.init(provider_uri=qlib_data_path, region=qlib_region)
    
    # Get calendar
    cal = D.calendar()
    if len(cal) == 0:
        print("\n⚠️  WARNING: No calendar data found! Please check your Qlib data path.")
        return
    
    print(f"\nCalendar range: {cal[0].date()} to {cal[-1].date()}")
    
    # Check if symbol exists in instruments
    try:
        instruments = D.instruments('all')
        if symbol not in instruments:
            print(f"\n⚠️  WARNING: Symbol '{symbol}' not found in Qlib instruments list.")
            print(f"   This doesn't necessarily mean data doesn't exist - checking files directly...")
    except Exception as e:
        print(f"\n⚠️  Could not check instruments list: {e}")
        print(f"   Checking files directly...")
    
    # Check feature files directly (Qlib stores data as binary files in symbol directories)
    expanded_path = os.path.expanduser(qlib_data_path)
    features_path = os.path.join(expanded_path, "features")
    
    if not os.path.exists(features_path):
        print(f"\n❌ Features directory not found: {features_path}")
        return
    
    # Qlib stores features in symbol directories with .bin files
    # Format: features/{symbol_lowercase}/{feature}.day.bin
    symbol_dir = os.path.join(features_path, symbol.lower())
    
    print(f"\n{'='*60}")
    print(f"Checking feature files:")
    print(f"{'='*60}")
    
    if not os.path.exists(symbol_dir):
        print(f"  ✗ Symbol directory not found: {symbol_dir}")
        print(f"     (Symbol might not be in Qlib data)")
    else:
        print(f"  ✓ Symbol directory found: {symbol_dir}")
        
        # Check for common feature files
        feature_files = {
            'close': 'close.day.bin',
            'open': 'open.day.bin', 
            'high': 'high.day.bin',
            'low': 'low.day.bin',
            'volume': 'volume.day.bin'
        }
        
        found_features = []
        missing_features = []
        
        for feature_name, filename in feature_files.items():
            feature_path = os.path.join(symbol_dir, filename)
            if os.path.exists(feature_path):
                found_features.append(feature_name)
                file_size = os.path.getsize(feature_path)
                print(f"  ✓ {feature_name}: Found ({file_size:,} bytes)")
            else:
                missing_features.append(feature_name)
                print(f"  ✗ {feature_name}: Not found")
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"{'='*60}")
        print(f"Found features: {len(found_features)}/{len(feature_files)}")
        if missing_features:
            print(f"Missing features: {', '.join(missing_features)}")
    
    # Try to load data using Qlib's data loader
    print(f"\n{'='*60}")
    print(f"Attempting to load data via Qlib:")
    print(f"{'='*60}")
    
    try:
        data_fields = ['$close', '$open', '$high', '$low', '$volume']
        data_loader = QlibDataLoader(config=data_fields)
        
        # Try loading from calendar start to end
        start_time = cal[0].strftime('%Y-%m-%d')
        end_time = cal[-1].strftime('%Y-%m-%d')
        
        data_df = data_loader.load(symbol, start_time, end_time)
        
        if data_df.empty:
            print(f"  ⚠️  No data loaded for {symbol}")
        else:
            # Get date range from loaded data
            symbol_data = data_df[symbol]
            if hasattr(symbol_data, 'index'):
                dates = symbol_data.index.get_level_values('datetime').unique()
                if len(dates) > 0:
                    min_date = dates.min()
                    max_date = dates.max()
                    print(f"  ✓ Successfully loaded data for {symbol}")
                    print(f"    Date range: {min_date.date()} to {max_date.date()}")
                    print(f"    Total data points: {len(dates)}")
                    
                    # Check which features are available
                    available_features = []
                    for field in data_fields:
                        field_data = symbol_data.xs(field.replace('$', ''), level='field', drop_level=False)
                        if not field_data.empty:
                            available_features.append(field)
                    print(f"    Available features: {', '.join(available_features)}")
                else:
                    print(f"  ⚠️  Data loaded but no dates found")
            else:
                print(f"  ⚠️  Data loaded but unexpected format")
    except Exception as e:
        print(f"  ✗ Error loading data via Qlib: {e}")
        print(f"    This might mean the symbol is not available in Qlib's data")
    
    print(f"{'='*60}")


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Check available time range in Qlib data, or check data for a specific symbol"
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=None,
        help="Check data availability for a specific symbol (e.g., AAPL, MSFT)"
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
    
    args = parser.parse_args()
    
    if args.symbol:
        # Check specific symbol
        check_symbol_data(args.symbol, args.qlib_path, args.region)
    else:
        # Check overall time range
        check_qlib_time_range()


if __name__ == '__main__':
    main()


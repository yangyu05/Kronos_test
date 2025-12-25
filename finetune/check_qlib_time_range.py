"""
Script to check the available time range in Qlib data.
"""
import pandas as pd
import qlib
from qlib.config import REG_US, REG_CN
from qlib.data import D
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

if __name__ == '__main__':
    check_qlib_time_range()


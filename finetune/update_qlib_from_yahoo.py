#!/usr/bin/env python3
"""
Script to update Qlib US market data from Yahoo Finance.

This script:
1. Downloads recent data (from 2020-11-11 onwards) from Yahoo Finance
2. Converts it to Qlib format
3. Updates the existing Qlib data directory

Usage:
    python finetune/update_qlib_from_yahoo.py [--symbols SYMBOL1 SYMBOL2 ...] [--start-date YYYY-MM-DD]
"""

import os
import sys
import pickle
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance not installed. Installing...")
    os.system(f"{sys.executable} -m pip install yfinance")
    import yfinance as yf

# Set up gymnasium as gym replacement BEFORE importing qlib
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    pass

import qlib
from qlib.config import REG_US
from qlib.data import D
from config import Config


class YahooToQlibUpdater:
    """Updates Qlib data from Yahoo Finance."""
    
    def __init__(self, qlib_data_path: str, start_date: str = "2020-11-11"):
        """
        Initialize the updater.
        
        Args:
            qlib_data_path: Path to Qlib data directory
            start_date: Start date for data collection (default: day after Qlib data ends)
        """
        self.qlib_data_path = os.path.expanduser(qlib_data_path)
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Qlib directory structure
        self.calendar_path = os.path.join(self.qlib_data_path, "calendars", "day.txt")
        self.instruments_path = os.path.join(self.qlib_data_path, "instruments", "all.txt")
        self.features_path = os.path.join(self.qlib_data_path, "features")
        
        # Ensure directories exist
        os.makedirs(os.path.join(self.qlib_data_path, "calendars"), exist_ok=True)
        os.makedirs(os.path.join(self.qlib_data_path, "instruments"), exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
        
        # Feature directories
        self.feature_dirs = {
            '$close': os.path.join(self.features_path, '$close', 'day'),
            '$open': os.path.join(self.features_path, '$open', 'day'),
            '$high': os.path.join(self.features_path, '$high', 'day'),
            '$low': os.path.join(self.features_path, '$low', 'day'),
            '$volume': os.path.join(self.features_path, '$volume', 'day'),
            '$vwap': os.path.join(self.features_path, '$vwap', 'day'),
        }
        
        for dir_path in self.feature_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def load_existing_instruments(self) -> dict:
        """Load existing instruments and their date ranges."""
        instruments = {}
        if os.path.exists(self.instruments_path):
            with open(self.instruments_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            symbol, start, end = parts[0], parts[1], parts[2]
                            instruments[symbol] = {'start': start, 'end': end}
        return instruments
    
    def load_existing_calendar(self) -> set:
        """Load existing calendar dates."""
        calendar_dates = set()
        if os.path.exists(self.calendar_path):
            with open(self.calendar_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        calendar_dates.add(line)
        return calendar_dates
    
    def download_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Download data for a symbol from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            # Download data from start_date to today
            df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            
            if df.empty:
                print(f"  ⚠️  No data for {symbol}")
                return None
            
            # Rename columns to match Qlib format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate VWAP (Volume Weighted Average Price)
            # VWAP = (High + Low + Close) / 3 * Volume / Volume
            # Simplified: use typical price * volume / volume
            df['vwap'] = ((df['high'] + df['low'] + df['close']) / 3)
            
            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
            for col in required_cols:
                if col not in df.columns:
                    print(f"  ⚠️  Missing column {col} for {symbol}")
                    return None
            
            # Remove any rows with NaN values
            df = df.dropna(subset=required_cols)
            
            if df.empty:
                print(f"  ⚠️  No valid data after cleaning for {symbol}")
                return None
            
            return df[required_cols]
            
        except Exception as e:
            print(f"  ❌ Error downloading {symbol}: {e}")
            return None
    
    def save_feature_file(self, symbol: str, feature: str, data: pd.Series):
        """
        Save a feature file in Qlib format.
        
        Note: Qlib uses .bin files, not .pkl. This is a simplified version.
        For production use, consider using Qlib's data collector script.
        
        Args:
            symbol: Stock symbol
            feature: Feature name (e.g., '$close')
            data: Pandas Series with dates as index
        """
        # Qlib stores data in: features/{symbol}/{feature}.day.bin
        # But the structure varies. For now, we'll use a workaround.
        # The proper way is to use Qlib's data collector script.
        
        # Create symbol directory
        symbol_dir = os.path.join(self.features_path, symbol.lower())
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Map feature names
        feature_map = {
            '$close': 'close',
            '$open': 'open',
            '$high': 'high',
            '$low': 'low',
            '$volume': 'volume',
            '$vwap': 'vwap',
        }
        
        feature_name = feature_map.get(feature, feature.replace('$', ''))
        file_path = os.path.join(symbol_dir, f"{feature_name}.day.bin")
        
        # Note: Qlib uses a specific binary format. 
        # This is a placeholder - you may need to use Qlib's data collector
        # or convert the data properly using Qlib's utilities
        print(f"    ⚠️  Note: Direct .bin file writing not implemented.")
        print(f"    💡 Consider using Qlib's data collector script instead.")
        print(f"    📝 Data would be saved to: {file_path}")
    
    def update_instrument(self, symbol: str, start_date: str, end_date: str):
        """Update or add instrument entry."""
        instruments = self.load_existing_instruments()
        
        if symbol in instruments:
            # Update end date if new data extends it
            existing_start = instruments[symbol]['start']
            existing_end = instruments[symbol]['end']
            
            # Use earliest start and latest end
            new_start = min(existing_start, start_date)
            new_end = max(existing_end, end_date)
            
            instruments[symbol] = {'start': new_start, 'end': new_end}
        else:
            # New instrument
            instruments[symbol] = {'start': start_date, 'end': end_date}
        
        # Write back to file
        with open(self.instruments_path, 'w') as f:
            for sym, dates in sorted(instruments.items()):
                f.write(f"{sym}\t{dates['start']}\t{dates['end']}\n")
    
    def update_calendar(self, dates: set):
        """Update calendar with new dates."""
        existing_dates = self.load_existing_calendar()
        all_dates = existing_dates.union(dates)
        
        # Write sorted dates
        sorted_dates = sorted([pd.Timestamp(d) for d in all_dates])
        with open(self.calendar_path, 'w') as f:
            for date in sorted_dates:
                f.write(f"{date.strftime('%Y-%m-%d')}\n")
    
    def update_symbol(self, symbol: str) -> bool:
        """
        Update data for a single symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Downloading {symbol}...")
        
        # Download data
        df = self.download_symbol_data(symbol)
        if df is None or df.empty:
            return False
        
        # Get date range
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        # Save feature files
        print(f"  💾 Saving features for {symbol} ({start_date} to {end_date})...")
        for feature, col_name in [
            ('$close', 'close'),
            ('$open', 'open'),
            ('$high', 'high'),
            ('$low', 'low'),
            ('$volume', 'volume'),
            ('$vwap', 'vwap'),
        ]:
            if col_name in df.columns:
                self.save_feature_file(symbol, feature, df[col_name])
        
        # Update instrument entry
        self.update_instrument(symbol, start_date, end_date)
        
        # Update calendar
        self.update_calendar(set(df.index.strftime('%Y-%m-%d')))
        
        print(f"  ✓ Updated {symbol}")
        return True
    
    def get_symbols_from_config(self) -> list:
        """Get symbols from config file."""
        config = Config()
        
        # Initialize Qlib to get instruments
        qlib.init(provider_uri=self.qlib_data_path, region=REG_US)
        
        if config.custom_symbols:
            return config.custom_symbols
        else:
            # Get symbols from the instrument
            try:
                instruments = D.instruments(config.instrument)
                return list(instruments)
            except Exception as e:
                print(f"⚠️  Could not get instruments: {e}")
                return []
    
    def update_all_symbols(self, symbols: list = None):
        """
        Update data for all symbols.
        
        Args:
            symbols: List of symbols to update. If None, uses symbols from config.
        """
        if symbols is None:
            print("📋 Getting symbols from config...")
            symbols = self.get_symbols_from_config()
        
        if not symbols:
            print("❌ No symbols to update!")
            return
        
        print(f"🔄 Updating {len(symbols)} symbols from {self.start_date} to {self.end_date}...")
        print("=" * 60)
        
        successful = 0
        failed = 0
        
        for symbol in tqdm(symbols, desc="Updating symbols"):
            if self.update_symbol(symbol):
                successful += 1
            else:
                failed += 1
        
        print("=" * 60)
        print(f"✅ Successfully updated: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"\n📊 Data updated in: {self.qlib_data_path}")


def check_qlib_collector():
    """Check if Qlib's data collector is available."""
    import qlib
    import os
    qlib_path = os.path.dirname(qlib.__file__)
    collector_path = os.path.join(qlib_path, 'scripts', 'data_collector', 'yahoo', 'collector.py')
    
    if os.path.exists(collector_path):
        return collector_path
    
    # Check if it's in the GitHub repo structure (common locations)
    possible_paths = [
        os.path.join(os.path.expanduser('~'), 'qlib', 'scripts', 'data_collector', 'yahoo', 'collector.py'),
        os.path.join(os.path.expanduser('~'), 'qlib_repo', 'scripts', 'data_collector', 'yahoo', 'collector.py'),
        os.path.join(os.path.expanduser('~'), 'Kronos_test', 'qlib_repo', 'scripts', 'data_collector', 'yahoo', 'collector.py'),
    ]
    
    for github_path in possible_paths:
        if os.path.exists(github_path):
            return github_path
    
    return None


def get_collector_source_dir(region='US'):
    """Get the source directory where Qlib collector stores downloaded CSV files."""
    # Default locations for Qlib collector
    collector_path = check_qlib_collector()
    
    possible_paths = [
        os.path.join(os.path.expanduser('~'), 'qlib', 'scripts', 'data_collector', 'yahoo', 'source'),
    ]
    
    if collector_path:
        possible_paths.insert(0, os.path.join(os.path.dirname(collector_path), 'source'))
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    
    # Return default path even if it doesn't exist yet
    return os.path.join(os.path.expanduser('~'), 'qlib', 'scripts', 'data_collector', 'yahoo', 'source')


def normalize_symbol_for_filename(symbol, region='US'):
    """Normalize symbol to match Qlib's filename format."""
    # Qlib uses code_to_fname to convert symbols to filenames
    # For US: usually uppercase, replace special chars
    symbol = symbol.upper()
    symbol = symbol.replace('.', '-')
    symbol = symbol.replace('^', '')
    return symbol


def check_downloaded_symbols(source_dir, requested_symbols=None, region='US'):
    """
    Check which symbols are already downloaded in the source directory.
    
    Args:
        source_dir: Path to source directory with CSV files
        requested_symbols: List of symbols to check (None = check all)
        region: Market region
        
    Returns:
        dict with 'downloaded' and 'missing' lists
    """
    if not source_dir:
        return {'downloaded': [], 'missing': requested_symbols or [], 'total_files': 0}
    
    if not os.path.exists(source_dir):
        return {'downloaded': [], 'missing': requested_symbols or [], 'total_files': 0}
    
    # Get all CSV files in source directory
    csv_files = [f.replace('.csv', '') for f in os.listdir(source_dir) if f.endswith('.csv')]
    downloaded_symbols = set(csv_files)
    total_files = len(csv_files)
    
    if requested_symbols is None:
        # Return all downloaded symbols
        return {'downloaded': list(downloaded_symbols), 'missing': [], 'total_files': total_files}
    
    # Normalize requested symbols for comparison
    normalized_requested = {normalize_symbol_for_filename(s, region): s for s in requested_symbols}
    normalized_downloaded = {normalize_symbol_for_filename(s, region): s for s in downloaded_symbols}
    
    # Check which are downloaded
    downloaded = []
    missing = []
    
    for norm_symbol, orig_symbol in normalized_requested.items():
        # Check if normalized version exists or if any downloaded file matches
        if norm_symbol in normalized_downloaded or any(
            norm_symbol in d or d in norm_symbol for d in downloaded_symbols
        ):
            downloaded.append(orig_symbol)
        else:
            missing.append(orig_symbol)
    
    return {'downloaded': downloaded, 'missing': missing, 'total_files': total_files}


def get_python_env():
    """
    Get environment variables with proper PYTHONPATH for subprocess calls.
    Handles both virtual environments and system Python.
    """
    env = os.environ.copy()
    python_paths = []
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a venv - add venv's site-packages
        venv_site = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        if os.path.exists(venv_site):
            python_paths.append(venv_site)
    
    # Also add user site-packages
    user_site = os.path.join(os.path.expanduser('~'), '.local', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    if os.path.exists(user_site):
        python_paths.append(user_site)
    
    # Update PYTHONPATH
    if python_paths:
        current_pythonpath = env.get('PYTHONPATH', '')
        new_paths = ':'.join(python_paths)
        env['PYTHONPATH'] = f"{new_paths}:{current_pythonpath}" if current_pythonpath else new_paths
    
    return env


def install_collector_dependencies():
    """Install dependencies needed for Qlib's collector."""
    print("Installing dependencies for Qlib collector...")
    dependencies = ['yahooquery', 'yfinance']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep} already installed")
        except ImportError:
            print(f"  📦 Installing {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
                print(f"  ✓ {dep} installed")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {dep}: {e}")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update Qlib data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Qlib uses a specific binary format (.bin files) for data storage.
This script provides a basic framework, but for production use, consider:

1. Using Qlib's official data collector:
   - Download from: https://github.com/microsoft/qlib/tree/main/scripts/data_collector
   - Run: python scripts/data_collector/yahoo/collector.py update_data_to_bin
   
2. Or use this script as a starting point and adapt it to Qlib's binary format.

For more info, see: finetune/UPDATE_YAHOO_DATA_README.md
        """
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        help='Specific symbols to update (e.g., AAPL MSFT GOOGL). If not provided, uses symbols from config.'
    )
    parser.add_argument(
        '--start-date',
        default='2020-11-11',
        help='Start date for data collection (default: 2020-11-11)'
    )
    parser.add_argument(
        '--qlib-path',
        default='~/.qlib/qlib_data/us_data',
        help='Path to Qlib data directory (default: ~/.qlib/qlib_data/us_data)'
    )
    parser.add_argument(
        '--use-qlib-collector',
        action='store_true',
        help='Try to use Qlib\'s official data collector if available'
    )
    parser.add_argument(
        '--region',
        default='US',
        choices=['US', 'CN', 'IN', 'BR'],
        help='Market region to download (default: US)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step and go directly to normalization (if source files exist)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if files already exist'
    )
    
    args = parser.parse_args()
    
    # Check for Qlib collector
    if args.use_qlib_collector:
        collector_path = check_qlib_collector()
        if collector_path:
            print("=" * 60)
            print("Using Qlib's Official Data Collector")
            print("=" * 60)
            print(f"Collector found at: {collector_path}")
            
            # Check for existing downloads
            source_dir = get_collector_source_dir(args.region)
            csv_count = 0
            if source_dir:
                if os.path.exists(source_dir):
                    csv_count = len([f for f in os.listdir(source_dir) if f.endswith('.csv')])
                    print(f"\n📁 Source directory: {source_dir}")
                    print(f"   Found {csv_count} existing CSV files")
                else:
                    print(f"\n📁 Source directory: {source_dir}")
                    print(f"   Directory does not exist yet (will be created during download)")
                
                # Check specific symbols if provided
                if args.symbols:
                    download_status = check_downloaded_symbols(source_dir, args.symbols, args.region)
                    if download_status['downloaded']:
                        print(f"\n✓ Already downloaded ({len(download_status['downloaded'])}/{len(args.symbols)}):")
                        for sym in download_status['downloaded'][:10]:  # Show first 10
                            print(f"   - {sym}")
                        if len(download_status['downloaded']) > 10:
                            print(f"   ... and {len(download_status['downloaded']) - 10} more")
                    
                    if download_status['missing']:
                        print(f"\n📥 Need to download ({len(download_status['missing'])}/{len(args.symbols)}):")
                        for sym in download_status['missing'][:10]:  # Show first 10
                            print(f"   - {sym}")
                        if len(download_status['missing']) > 10:
                            print(f"   ... and {len(download_status['missing']) - 10} more")
                    
                    # If all symbols are downloaded and skip_download is not set, ask or auto-skip
                    if not download_status['missing'] and not args.skip_download and not args.force_download:
                        print(f"\n✅ All requested symbols are already downloaded!")
                        print("   Skipping download step. Use --force-download to re-download.")
                        args.skip_download = True
                
                if args.skip_download:
                    print("\n⏭️  Skipping download step")
                    if csv_count > 0:
                        print("   Proceeding directly to normalization...")
                    else:
                        print("   ⚠️  Warning: No source files found. Normalization may fail.")
                elif csv_count > 0 and not args.force_download:
                    print(f"\n✓ Found existing downloaded data ({csv_count} files)")
                    print("   Collector will append new data to existing files")
                    print("   (Use --force-download to force re-download)")
            
            # Install dependencies
            if not install_collector_dependencies():
                print("⚠️  Some dependencies failed to install. Continuing anyway...")
            
            if not args.skip_download:
                print("\n📥 Running Qlib collector (download step)...")
                cmd = [
                    sys.executable,
                    collector_path,
                    'update_data_to_bin',
                    '--qlib_data_1d_dir', os.path.expanduser(args.qlib_path),
                    '--trading_date', args.start_date,
                    '--end_date', datetime.now().strftime('%Y-%m-%d'),
                    '--region', args.region.upper(),
                    '--interval', '1d'
                ]
                # Get environment with proper PYTHONPATH (includes venv if in one)
                env = get_python_env()
                try:
                    result = subprocess.run(cmd, check=False, env=env)
                    if result.returncode == 0:
                        print("\n✅ Data collection completed successfully!")
                    else:
                        print(f"\n⚠️  Collector exited with code {result.returncode}")
                        print("   Check the output above for errors.")
                        return
                except Exception as e:
                    print(f"\n❌ Error running collector: {e}")
                    return
            
            # Run normalization step
            if source_dir and os.path.exists(source_dir):
                print("\n📊 Running normalization step...")
                normalize_dir = os.path.join(os.path.dirname(source_dir), 'normalize')
                normalize_cmd = [
                    sys.executable,
                    collector_path,
                    '--source_dir', source_dir,
                    '--normalize_dir', normalize_dir,
                    '--region', args.region.upper(),
                    '--interval', '1d',
                    'normalize_data_1d_extend',
                    os.path.expanduser(args.qlib_path)  # Positional argument: old_qlib_data_dir
                ]
                # Get environment with proper PYTHONPATH (includes venv if in one)
                env = get_python_env()
                try:
                    result = subprocess.run(normalize_cmd, check=False, env=env)
                    if result.returncode == 0:
                        print("\n✅ Normalization completed successfully!")
                    else:
                        print(f"\n⚠️  Normalization exited with code {result.returncode}")
                except Exception as e:
                    print(f"\n❌ Error during normalization: {e}")
            else:
                print("\n⚠️  Source directory not found. Cannot run normalization.")
            
            return
        else:
            print("⚠️  Qlib collector not found. Using basic script instead.")
            print("   To get Qlib collector, run:")
            print("   python finetune/download_qlib_collector.py")
    
    print("=" * 60)
    print("Yahoo Finance to Qlib Data Updater")
    print("=" * 60)
    print("⚠️  NOTE: This is a basic implementation.")
    print("   Qlib uses .bin files in a specific format.")
    print("   For production use, consider Qlib's official collector.")
    print("=" * 60)
    print(f"Qlib path: {os.path.expanduser(args.qlib_path)}")
    print(f"Start date: {args.start_date}")
    print(f"End date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    updater = YahooToQlibUpdater(args.qlib_path, args.start_date)
    updater.update_all_symbols(args.symbols)
    
    print("\n✅ Update complete!")
    print("\nNext steps:")
    print("1. Update config.py: set dataset_end_time to a more recent date")
    print("2. Re-run qlib_data_preprocess.py to process the new data")
    print("3. Verify data range: python check_qlib_time_range.py")
    print("\n💡 For better results, use Qlib's official data collector:")
    print("   See: finetune/UPDATE_YAHOO_DATA_README.md")


if __name__ == '__main__':
    main()


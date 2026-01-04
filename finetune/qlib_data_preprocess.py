import os
import pickle
import numpy as np
import pandas as pd

# Set up gymnasium as gym replacement BEFORE importing qlib
import sys
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
except ImportError:
    pass  # Fall back to gym if gymnasium not available

import qlib
from qlib.config import REG_CN, REG_US
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange

from config import Config


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.data = {}  # A dictionary to store processed data for each symbol.

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib...")
        # Use REG_US for US market, REG_CN for Chinese market
        region = REG_US if self.config.market_region == 'us' else REG_CN
        print(f"Using region: {self.config.market_region} ({region})")
        qlib.init(provider_uri=self.config.qlib_data_path, region=region)

    def load_qlib_data(self):
        """
        Loads raw data from Qlib, processes it symbol by symbol, and stores
        it in the `self.data` attribute.
        """
        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        
        # Get calendar with appropriate frequency
        data_freq = getattr(self.config, 'data_freq', 'day')
        print(f"Loading data with frequency: {data_freq}")
        cal: np.ndarray = D.calendar(freq=data_freq) if data_freq != 'day' else D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # Check if start_index lookbackw_window will cause negative index
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = cal[adjusted_start_index]

        # Check if end_index exceeds the range of the array
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # Check if end_index+predictw_window will exceed the range of the array
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = cal[adjusted_end_index]

        # Load data using Qlib's data loader with appropriate frequency
        data_freq = getattr(self.config, 'data_freq', 'day')
        
        if data_freq == 'day':
            # Use QlibDataLoader for daily data (original method)
            data_df = QlibDataLoader(config=data_fields_qlib).load(
                self.config.instrument, real_start_time, real_end_time
            )
            data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.
        else:
            # Use D.features for intraday frequencies (5min, 15min, etc.)
            print(f"Loading {data_freq} frequency data using D.features...")
            # For 5min data, D.instruments() doesn't work correctly, so read instruments file directly
            import os
            from pathlib import Path
            qlib_data_path = Path(os.path.expanduser(self.config.qlib_data_path))
            inst_file = qlib_data_path / "instruments" / f"{self.config.instrument}.txt"
            
            if inst_file.exists():
                # Read symbols from instruments file
                instruments = []
                with open(inst_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) > 0:
                            symbol = parts[0].strip()
                            if symbol and symbol not in ['market', 'filter_pipe']:
                                instruments.append(symbol)
                print(f"Read {len(instruments)} instruments from {inst_file}")
            else:
                # Fallback to D.instruments
                instruments = list(D.instruments(self.config.instrument))
                instruments = [inst for inst in instruments if inst not in ['market', 'filter_pipe']]
                print(f"Using D.instruments (fallback): {len(instruments)} instruments")
            
            if len(instruments) == 0:
                raise ValueError(f"No instruments found for {self.config.instrument} in {qlib_data_path}")
            
            # Load data - try with just date part first (5min data might need date-only format)
            start_time_str = real_start_time.strftime('%Y-%m-%d') if hasattr(real_start_time, 'strftime') else str(real_start_time)
            end_time_str = real_end_time.strftime('%Y-%m-%d') if hasattr(real_end_time, 'strftime') else str(real_end_time)
            
            print(f"Loading data from {start_time_str} to {end_time_str} for {len(instruments)} instruments...")
            data_df = D.features(
                instruments,
                data_fields_qlib,
                start_time=start_time_str,
                end_time=end_time_str,
                freq=data_freq
            )
            print(f"Loaded DataFrame shape: {data_df.shape}")
            if len(data_df) > 0:
                print(f"  Index levels: {data_df.index.names if hasattr(data_df.index, 'names') else 'N/A'}")
                if isinstance(data_df.index, pd.MultiIndex):
                    unique_instruments = data_df.index.get_level_values('instrument').unique()
                    print(f"  Unique instruments in data: {len(unique_instruments)}")
                    print(f"  First few: {list(unique_instruments[:5])}")
            else:
                print("  WARNING: Loaded DataFrame is empty!")
            # For D.features, the structure is:
            # - MultiIndex on rows: (instrument, datetime)
            # - Columns: field names (e.g., '$close', '$open')
            # So fields are already columns, we just need to clean up column names
            # Check if columns are MultiIndex or regular Index
            if isinstance(data_df.columns, pd.MultiIndex):
                # If MultiIndex columns, we need to handle differently
                print(f"  WARNING: DataFrame has MultiIndex columns: {data_df.columns.names}")
                print(f"  Column structure: {data_df.columns[:5]}")
                # Flatten MultiIndex columns if needed
                if data_df.columns.nlevels > 1:
                    data_df.columns = [col[-1] if isinstance(col, tuple) else str(col) for col in data_df.columns]
            # Clean up column names (remove $ prefix)
            data_df.columns = [col.replace('$', '') for col in data_df.columns]
            print(f"  Final columns after cleanup: {list(data_df.columns[:10])}")

        # Process symbols based on data structure
        if data_freq == 'day':
            # Daily data: symbols are columns
            symbol_list = list(data_df.columns)
        else:
            # Intraday data: symbols are in MultiIndex (instrument, datetime)
            # Extract unique instruments from index
            if isinstance(data_df.index, pd.MultiIndex) and 'instrument' in data_df.index.names:
                symbol_list = data_df.index.get_level_values('instrument').unique().tolist()
                # Filter out system instruments like 'market', 'filter_pipe'
                symbol_list = [s for s in symbol_list if s not in ['market', 'filter_pipe']]
            else:
                # Fallback: try columns (shouldn't happen for D.features)
                symbol_list = list(data_df.columns)
        
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            
            if data_freq == 'day':
                # Daily data processing
                symbol_df = data_df[symbol]
                # Pivot the table to have features as columns and datetime as index.
                symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
                symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
                symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})
            else:
                # Intraday data processing
                # Extract data for this symbol using xs (cross-section)
                try:
                    symbol_df = data_df.xs(symbol, level='instrument')
                except KeyError:
                    if i < 5:  # Debug first few
                        print(f"  KeyError extracting {symbol}")
                    continue
                
                # symbol_df now has datetime as index and fields as columns
                # Ensure datetime index is properly set
                if not isinstance(symbol_df.index, pd.DatetimeIndex):
                    symbol_df.index = pd.to_datetime(symbol_df.index, errors='coerce')
                    symbol_df = symbol_df[~symbol_df.index.isna()]
                
                # Make a copy to avoid SettingWithCopyWarning and ensure we have our own DataFrame
                symbol_df = symbol_df.copy()
                
                # Debug first few symbols
                if i < 3:
                    print(f"  {symbol}: {len(symbol_df)} rows after extraction, columns: {list(symbol_df.columns)}")
                    print(f"    NaN counts: {symbol_df.isna().sum().to_dict()}")
            
            # Calculate amount and select final features (common for both daily and intraday)
            initial_len = len(symbol_df)
            if 'volume' in symbol_df.columns:
                symbol_df['vol'] = symbol_df['volume']
            elif 'vol' not in symbol_df.columns:
                # If volume is missing, skip this symbol
                if i < 5:
                    print(f"  {symbol}: missing volume column")
                continue
            
            # Calculate amount if needed
            if 'amt' not in symbol_df.columns and all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close', 'vol']):
                symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            
            # Select only features that exist
            available_features = [f for f in self.config.feature_list if f in symbol_df.columns]
            if len(available_features) < len(self.config.feature_list):
                if i < 5:
                    print(f"  {symbol}: missing features: {set(self.config.feature_list) - set(available_features)}")
            
            symbol_df = symbol_df[available_features]
            after_feature_selection = len(symbol_df)

            # Filter out symbols with insufficient data.
            symbol_df = symbol_df.dropna()
            after_dropna = len(symbol_df)
            min_required = self.config.lookback_window + self.config.predict_window + 1
            
            if i < 5:
                print(f"  {symbol}: {initial_len} -> {after_feature_selection} -> {after_dropna} (need {min_required})")
            
            if len(symbol_df) < min_required:
                if i < 5:
                    print(f"  Skipping {symbol}: only {len(symbol_df)} data points (need {min_required})")
                continue

            self.data[symbol] = symbol_df

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Define time ranges from config.
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # Convert string dates to Timestamps for proper comparison
            train_start_ts = pd.Timestamp(train_start)
            train_end_ts = pd.Timestamp(train_end)
            val_start_ts = pd.Timestamp(val_start)
            val_end_ts = pd.Timestamp(val_end)
            test_start_ts = pd.Timestamp(test_start)
            test_end_ts = pd.Timestamp(test_end)

            # Create boolean masks for each dataset split.
            train_mask = (symbol_df.index >= train_start_ts) & (symbol_df.index <= train_end_ts)
            val_mask = (symbol_df.index >= val_start_ts) & (symbol_df.index <= val_end_ts)
            test_mask = (symbol_df.index >= test_start_ts) & (symbol_df.index <= test_end_ts)

            # Apply masks to create the final datasets.
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()


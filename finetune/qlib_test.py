import os
import sys
import argparse
import pickle
from collections import defaultdict

# Set up gymnasium as gym replacement BEFORE importing qlib
# This must be done before any qlib imports since qlib imports gym internally
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
    print("✓ Using gymnasium as gym replacement")
except ImportError:
    print("⚠ Warning: gymnasium not found, falling back to gym (may have NumPy 2.0 issues)")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

import qlib
from qlib.config import REG_CN, REG_US
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.backtest import backtest, executor, CommonInfrastructure
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import flatten_dict
from qlib.utils.time import Freq

# Ensure project root is in the Python path
sys.path.append("../")
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


# =================================================================================
# 1. Data Loading and Processing for Inference
# =================================================================================

class QlibTestDataset(Dataset):
    """
    PyTorch Dataset for handling Qlib test data, specifically for inference.

    This dataset iterates through all possible sliding windows sequentially. It also
    yields metadata like symbol and timestamp, which are crucial for mapping
    predictions back to the original time series.
    """

    def __init__(self, data: dict, config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(self.data.keys())
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices = []

        print("Preprocessing and building indices for test dataset...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            # Generate time features on-the-fly
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df  # Store preprocessed dataframe

            num_samples = len(df) - self.window_size + 1
            if num_samples > 0:
                for i in range(num_samples):
                    timestamp = df.iloc[i + self.config.lookback_window - 1]['datetime']
                    self.indices.append((symbol, i, timestamp))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]

        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        # Instance-level normalization, consistent with training
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp), torch.from_numpy(y_stamp), symbol, timestamp


# =================================================================================
# 2. Backtesting Logic
# =================================================================================

class QlibBacktest:
    """
    A wrapper class for conducting backtesting experiments using Qlib.
    """

    def __init__(self, config: Config):
        self.config = config
        self.initialize_qlib()

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib for backtesting...")
        region = REG_US if self.config.market_region == 'us' else REG_CN
        qlib.init(provider_uri=self.config.qlib_data_path, region=region)

    def run_single_backtest(self, signal_series: pd.Series, benchmark: str = None, end_time: str = None) -> pd.DataFrame:
        """
        Runs a single backtest for a given prediction signal.

        Args:
            signal_series (pd.Series): A pandas Series with a MultiIndex
                                       (instrument, datetime) and prediction scores.
            benchmark (str, optional): Benchmark symbol to use. If None, uses config default.
            end_time (str, optional): End date for backtest (YYYY-MM-DD). If None, uses latest available from Qlib.
        Returns:
            pd.DataFrame: A DataFrame containing the performance report.
        """
        strategy = TopkDropoutStrategy(
            topk=self.config.backtest_n_symbol_hold,
            n_drop=self.config.backtest_n_symbol_drop,
            hold_thresh=self.config.backtest_hold_thresh,
            signal=signal_series,
        )
        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "delay_execution": True,
        }
        
        # Get available calendar and determine end_time
        # Qlib's backtest accesses calendar[index + 1], so we need at least one day buffer
        cal = D.calendar()
        if len(cal) == 0:
            raise ValueError("No calendar data available in Qlib!")
        
        # Determine end_time: use provided, or latest available, or config default
        if end_time is not None:
            requested_end_time = pd.Timestamp(end_time)
            # Use the second-to-last date to ensure calendar[index + 1] is valid
            safe_end_date = pd.Timestamp(cal[-2]) if len(cal) > 1 else pd.Timestamp(cal[-1])
            safe_end_time = min(requested_end_time, safe_end_date)
            if safe_end_time < requested_end_time:
                print(f"⚠️  Requested end_time {end_time} exceeds available data. Using {safe_end_time.date()}")
            end_time_str = safe_end_time.strftime('%Y-%m-%d')
        elif len(cal) > 1:
            # Use the latest available date (second-to-last for safety)
            safe_end_date = pd.Timestamp(cal[-2])
            configured_end_time = pd.Timestamp(self.config.backtest_time_range[1])
            # Use the later of configured end time or the safe end date (to use latest available)
            safe_end_time = max(configured_end_time, safe_end_date)
            if safe_end_time > configured_end_time:
                print(f"ℹ️  Using latest available date from Qlib: {safe_end_time.date()} (config had {configured_end_time.date()})")
            end_time_str = safe_end_time.strftime('%Y-%m-%d')
        else:
            end_time_str = self.config.backtest_time_range[1]
        
        # Use provided benchmark or fall back to config default
        benchmark_symbol = benchmark if benchmark is not None else self.config.backtest_benchmark
        
        backtest_config = {
            "start_time": self.config.backtest_time_range[0],
            "end_time": end_time_str,
            "account": 100_000_000,
            "benchmark": benchmark_symbol,
            "exchange_kwargs": {
                "freq": "day", "limit_threshold": 0.095, "deal_price": "open",
                "open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5,
            },
            "executor": executor.SimulatorExecutor(**executor_config),
        }

        portfolio_metric_dict, _ = backtest(strategy=strategy, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse("day"))
        report, _ = portfolio_metric_dict.get(analysis_freq)

        # --- Analysis and Reporting ---
        analysis = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], freq=analysis_freq),
            "excess_return_with_cost": risk_analysis(report["return"] - report["bench"] - report["cost"], freq=analysis_freq),
        }
        print("\n--- Backtest Analysis ---")
        print("Benchmark Return:", risk_analysis(report["bench"], freq=analysis_freq), sep='\n')
        print("\nExcess Return (w/o cost):", analysis["excess_return_without_cost"], sep='\n')
        print("\nExcess Return (w/ cost):", analysis["excess_return_with_cost"], sep='\n')

        report_df = pd.DataFrame({
            "cum_bench": report["bench"].cumsum(),
            "cum_return_w_cost": (report["return"] - report["cost"]).cumsum(),
            "cum_ex_return_w_cost": (report["return"] - report["bench"] - report["cost"]).cumsum(),
        })
        return report_df

    def run_and_plot_results(self, signals: dict[str, pd.DataFrame], benchmark: str = None, end_time: str = None):
        """
        Runs backtests for multiple signals and plots the cumulative return curves.

        Args:
            signals (dict[str, pd.DataFrame]): A dictionary where keys are signal names
                                               and values are prediction DataFrames.
            benchmark (str, optional): Benchmark symbol to use. If None, uses config default.
            end_time (str, optional): End date for backtest (YYYY-MM-DD). If None, uses latest available from Qlib.
        """
        return_df, ex_return_df, bench_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for signal_name, pred_df in signals.items():
            print(f"\nBacktesting signal: {signal_name}...")
            pred_series = pred_df.stack()
            pred_series.index.names = ['datetime', 'instrument']
            pred_series = pred_series.swaplevel().sort_index()
            report_df = self.run_single_backtest(pred_series, benchmark=benchmark, end_time=end_time)

            return_df[signal_name] = report_df['cum_return_w_cost']
            ex_return_df[signal_name] = report_df['cum_ex_return_w_cost']
            if 'return' not in bench_df:
                bench_df['return'] = report_df['cum_bench']

        # Determine benchmark label for plot
        benchmark_label = benchmark if benchmark is not None else self.config.backtest_benchmark

        # Plotting results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        return_df.plot(ax=axes[0], title='Cumulative Return with Cost', grid=True)
        axes[0].plot(bench_df['return'], label=f'Benchmark ({benchmark_label})', color='black', linestyle='--')
        axes[0].legend()
        axes[0].set_ylabel("Cumulative Return")

        ex_return_df.plot(ax=axes[1], title='Cumulative Excess Return with Cost', grid=True)
        axes[1].legend()
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Cumulative Excess Return")

        plt.tight_layout()
        plt.savefig("../figures/backtest_result_example.png", dpi=200)
        plt.show()


# =================================================================================
# 3. Data Loading from Qlib (for extending beyond preprocessed data)
# =================================================================================

def load_data_from_qlib(config: Config, symbols: list, start_time: str = None, end_time: str = None) -> dict:
    """
    Load data directly from Qlib for specified symbols.
    
    Args:
        config: Config object
        symbols: List of symbols to load
        start_time: Start date (YYYY-MM-DD). If None, uses config.test_time_range[0]
        end_time: End date (YYYY-MM-DD). If None, uses latest available date from Qlib
    
    Returns:
        Dictionary mapping symbol to DataFrame with datetime index and feature columns
    """
    # Initialize Qlib
    region = REG_US if config.market_region == 'us' else REG_CN
    qlib.init(provider_uri=config.qlib_data_path, region=region)
    
    # Get calendar to determine available dates
    cal = D.calendar()
    if len(cal) == 0:
        raise ValueError("No calendar data found in Qlib!")
    
    # Determine time range
    if start_time is None:
        start_time = config.test_time_range[0]
    if end_time is None:
        # Use latest available date (with buffer for predict_window)
        latest_date = cal[-1]
        end_time = latest_date.strftime('%Y-%m-%d')
        print(f"Using latest available date from Qlib: {end_time}")
    else:
        # Ensure end_time doesn't exceed available data
        end_time_ts = pd.Timestamp(end_time)
        if end_time_ts > cal[-1]:
            end_time = cal[-1].strftime('%Y-%m-%d')
            print(f"⚠️  Requested end_time exceeds available data. Using latest available: {end_time}")
    
    # Load data fields
    data_fields_qlib = ['$' + f for f in config.feature_list]
    
    # Load data for each symbol
    data_dict = {}
    print(f"\nLoading data from Qlib for {len(symbols)} symbol(s) from {start_time} to {end_time}...")
    
    for symbol in tqdm(symbols, desc="Loading symbols"):
        try:
            # Load data for this symbol
            data_df = QlibDataLoader(config=data_fields_qlib).load(
                symbol, start_time, end_time
            )
            
            if data_df.empty:
                print(f"  ⚠️  {symbol}: No data found")
                continue
            
            # Reshape data
            symbol_df = data_df[symbol].unstack(level=1)
            symbol_df.columns = [col.replace('$', '') for col in symbol_df.columns]
            
            # Calculate amount if needed
            if 'amt' in config.feature_list and 'amt' not in symbol_df.columns:
                if all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close', 'vol']):
                    symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            
            # Select only requested features
            available_features = [f for f in config.feature_list if f in symbol_df.columns]
            symbol_df = symbol_df[available_features]
            
            # Remove rows with any NaN values
            symbol_df = symbol_df.dropna()
            
            if len(symbol_df) > 0:
                data_dict[symbol] = symbol_df
                print(f"  ✓ {symbol}: {len(symbol_df)} data points ({symbol_df.index.min().date()} to {symbol_df.index.max().date()})")
            else:
                print(f"  ⚠️  {symbol}: No valid data after cleaning")
        except Exception as e:
            print(f"  ✗ {symbol}: Error loading data - {e}")
    
    return data_dict


# =================================================================================
# 4. Inference Logic
# =================================================================================

def load_models(config: dict) -> tuple[KronosTokenizer, Kronos]:
    """Loads the fine-tuned tokenizer and predictor model."""
    device = torch.device(config['device'])
    print(f"Loading models onto device: {device}...")
    
    # Convert relative paths to absolute paths
    tokenizer_path = config['tokenizer_path']
    model_path = config['model_path']
    
    if not os.path.isabs(tokenizer_path):
        # Get the project root (parent of finetune directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        tokenizer_path = os.path.join(project_root, tokenizer_path)
    
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, model_path)
    
    print(f"Loading weights from local directory")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device).eval()
    print(f"Loading weights from local directory")
    model = Kronos.from_pretrained(model_path).to(device).eval()
    return tokenizer, model


def collate_fn_for_inference(batch):
    """
    Custom collate function to handle batches containing Tensors, strings, and Timestamps.

    Args:
        batch (list): A list of samples, where each sample is the tuple returned by
                      QlibTestDataset.__getitem__.

    Returns:
        A single tuple containing the batched data.
    """
    # Unzip the list of samples into separate lists for each data type
    x, x_stamp, y_stamp, symbols, timestamps = zip(*batch)

    # Stack the tensors to create a batch
    x_batch = torch.stack(x, dim=0)
    x_stamp_batch = torch.stack(x_stamp, dim=0)
    y_stamp_batch = torch.stack(y_stamp, dim=0)

    # Return the strings and timestamps as lists
    return x_batch, x_stamp_batch, y_stamp_batch, list(symbols), list(timestamps)


def generate_predictions(config: dict, test_data: dict) -> dict[str, pd.DataFrame]:
    """
    Runs inference on the test dataset to generate prediction signals.

    Args:
        config (dict): A dictionary containing inference parameters.
        test_data (dict): The raw test data loaded from a pickle file.

    Returns:
        A dictionary where keys are signal types (e.g., 'mean', 'last') and
        values are DataFrames of predictions (datetime index, symbol columns).
    """
    tokenizer, model = load_models(config)
    device = torch.device(config['device'])

    # Use the Dataset and DataLoader for efficient batching and processing
    dataset = QlibTestDataset(data=test_data, config=Config())
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'] // config['sample_count'],
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=collate_fn_for_inference
    )

    results = defaultdict(list)
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps in tqdm(loader, desc="Inference"):
            preds = auto_regressive_inference(
                tokenizer, model, x.to(device), x_stamp.to(device), y_stamp.to(device),
                max_context=config['max_context'], pred_len=config['pred_len'], clip=config['clip'],
                T=config['T'], top_k=config['top_k'], top_p=config['top_p'], sample_count=config['sample_count']
            )
            # You can try commenting on this line to keep the history data
            preds = preds[:, -config['pred_len']:, :]

            # The 'close' price is at index 3 in `feature_list`
            last_day_close = x[:, -1, 3].numpy()
            signals = {
                'last': preds[:, -1, 3] - last_day_close,
                'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,
                'max': np.max(preds[:, :, 3], axis=1) - last_day_close,
                'min': np.min(preds[:, :, 3], axis=1) - last_day_close,
            }

            for i in range(len(symbols)):
                for sig_type, sig_values in signals.items():
                    results[sig_type].append((timestamps[i], symbols[i], sig_values[i]))

    print("Post-processing predictions into DataFrames...")
    prediction_dfs = {}
    for sig_type, records in results.items():
        df = pd.DataFrame(records, columns=['datetime', 'instrument', 'score'])
        pivot_df = df.pivot_table(index='datetime', columns='instrument', values='score')
        prediction_dfs[sig_type] = pivot_df.sort_index()

    return prediction_dfs


# =================================================================================
# 5. Main Execution
# =================================================================================

def main():
    """Main function to set up config, run inference, and execute backtesting."""
    parser = argparse.ArgumentParser(description="Run Kronos Inference and Backtesting")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for inference (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--symbols", type=str, nargs='+', default=['QQQ', 'DIA', 'SPY'],
                        help="Symbols to focus on (default: QQQ DIA SPY). Note: If ETFs not available, use stocks like: AAPL MSFT GOOGL")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Benchmark symbol to use for comparison (e.g., 'AAPL', 'SPY'). If not provided, uses the first symbol from --symbols if only one symbol is provided, otherwise uses config default.")
    parser.add_argument("--reload-from-qlib", action='store_true',
                        help="Reload data directly from Qlib instead of using preprocessed pickle file. This allows using the latest available data.")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for backtesting (YYYY-MM-DD). If not provided, uses the latest available date from Qlib.")
    args = parser.parse_args()

    # --- 1. Configuration Setup ---
    base_config = Config()

    # Create a dedicated dictionary for this run's configuration
    run_config = {
        'device': args.device,
        'data_path': base_config.dataset_path,
        'result_save_path': base_config.backtest_result_path,
        'result_name': base_config.backtest_save_folder_name,
        'tokenizer_path': base_config.finetuned_tokenizer_path,
        'model_path': base_config.finetuned_predictor_path,
        'max_context': base_config.max_context,
        'pred_len': base_config.predict_window,
        'clip': base_config.clip,
        'T': base_config.inference_T,
        'top_k': base_config.inference_top_k,
        'top_p': base_config.inference_top_p,
        'sample_count': base_config.inference_sample_count,
        'batch_size': base_config.backtest_batch_size,
    }

    print("--- Running with Configuration ---")
    for key, val in run_config.items():
        print(f"{key:>20}: {val}")
    print("-" * 35)

    # --- 2. Load Data ---
    target_symbols = args.symbols
    
    if args.reload_from_qlib:
        # Load data directly from Qlib (allows using latest available data)
        print("\n🔄 Loading data directly from Qlib (bypassing preprocessed pickle file)...")
        test_data = load_data_from_qlib(base_config, target_symbols, 
                                        start_time=base_config.test_time_range[0],
                                        end_time=args.end_date)
        
        if len(test_data) == 0:
            print("\n❌ ERROR: No data could be loaded from Qlib for the specified symbols!")
            print("Please check:")
            print("  1. Qlib data path is correct")
            print("  2. Symbols are valid and available in Qlib")
            print("  3. Date range is valid")
            return
        
        filtered_test_data = test_data
    else:
        # Load from preprocessed pickle file
        test_data_path = os.path.join(run_config['data_path'], "test_data.pkl")
        print(f"\nLoading test data from {test_data_path}...")
        
        if not os.path.exists(test_data_path):
            print(f"❌ ERROR: Test data file not found: {test_data_path}")
            print("\n💡 Tip: Use --reload-from-qlib to load data directly from Qlib")
            return
        
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Filter to focus on specified symbols
        print(f"\nFiltering test data to focus on key symbols: {target_symbols}")
        filtered_test_data = {}
        for symbol in target_symbols:
            if symbol in test_data and len(test_data[symbol]) > 0:
                filtered_test_data[symbol] = test_data[symbol]
                print(f"  ✓ {symbol}: {len(test_data[symbol])} data points")
            else:
                print(f"  ✗ {symbol}: Not found in test data")
        
        if len(filtered_test_data) == 0:
            print("\n⚠️  WARNING: None of the target symbols were found in test data!")
            print(f"Available symbols (first 50): {list(test_data.keys())[:50]}")
            print("\nNote: QQQ, DIA, SPY are ETFs and may not be in S&P 500 stock dataset.")
            print("Try using major stocks like: AAPL MSFT GOOGL AMZN TSLA META")
            print("\n💡 Tip: Use --reload-from-qlib to load data directly from Qlib")
            return
    
    print(f"\nUsing {len(filtered_test_data)} symbol(s) for backtesting")
    test_data = filtered_test_data
    
    # Check date range of loaded data
    all_dates = []
    for symbol, df in filtered_test_data.items():
        if len(df) > 0:
            all_dates.extend(df.index.tolist())
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        print(f"Data date range: {min_date.date()} to {max_date.date()}")
    
    # Determine benchmark: use --benchmark if provided, otherwise use first symbol if only one symbol, else use config default
    if args.benchmark:
        benchmark_symbol = args.benchmark
        print(f"\nUsing specified benchmark: {benchmark_symbol}")
    elif len(filtered_test_data) == 1:
        benchmark_symbol = list(filtered_test_data.keys())[0]
        print(f"\nUsing single symbol as benchmark: {benchmark_symbol}")
    else:
        benchmark_symbol = None  # Will use config default
        print(f"\nUsing config default benchmark: {base_config.backtest_benchmark}")
    
    # --- 3. Generate Predictions ---
    model_preds = generate_predictions(run_config, test_data)

    # --- 4. Save Predictions ---
    save_dir = os.path.join(run_config['result_save_path'], run_config['result_name'])
    os.makedirs(save_dir, exist_ok=True)
    predictions_file = os.path.join(save_dir, "predictions.pkl")
    print(f"Saving prediction signals to {predictions_file}...")
    with open(predictions_file, 'wb') as f:
        pickle.dump(model_preds, f)

    # --- 5. Run Backtesting ---
    with open(predictions_file, 'rb') as f:
        model_preds = pickle.load(f)

    # Determine backtest end time
    backtest_end_time = args.end_date
    
    backtester = QlibBacktest(base_config)
    backtester.run_and_plot_results(model_preds, benchmark=benchmark_symbol, end_time=backtest_end_time)


if __name__ == '__main__':
    main()



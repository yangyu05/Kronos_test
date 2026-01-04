import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, input_len, plot_path="prediction_result.png"):
    """
    Plot predictions vs ground truth, removing market closed periods.
    
    Args:
        kline_df: DataFrame containing both input period and prediction period (ground truth)
        pred_df: DataFrame containing predictions
        input_len: Number of rows in kline_df that are input (not prediction period)
    """
    # Separate input period from prediction period
    input_df = kline_df.iloc[:input_len].copy()
    ground_truth_df = kline_df.iloc[input_len:].copy()
    
    # Ensure pred_df has the same index as ground_truth_df for alignment
    if len(pred_df) != len(ground_truth_df):
        print(f"⚠️  Warning: Prediction length ({len(pred_df)}) doesn't match ground truth length ({len(ground_truth_df)})")
        min_len = min(len(pred_df), len(ground_truth_df))
        pred_df = pred_df.iloc[:min_len]
        ground_truth_df = ground_truth_df.iloc[:min_len]
    
    # Combine all timestamps to create a continuous sequential index
    # This removes gaps from market closed periods (overnight, weekends, holidays)
    all_timestamps = pd.concat([input_df['timestamps'], ground_truth_df['timestamps']]).sort_values().reset_index(drop=True)
    
    # Create a mapping from timestamp to sequential index (0, 1, 2, ...)
    # This makes the plot continuous without showing market closed periods
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(all_timestamps)}
    
    # Map timestamps to sequential indices
    input_seq_idx = [timestamp_to_idx[ts] for ts in input_df['timestamps']]
    pred_seq_idx = [timestamp_to_idx[ts] for ts in ground_truth_df['timestamps']]
    
    # Get values
    input_close = input_df['close'].values
    input_volume = input_df['volume'].values
    gt_close = ground_truth_df['close'].values
    gt_volume = ground_truth_df['volume'].values
    pred_close = pred_df['close'].values
    pred_volume = pred_df['volume'].values
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot close prices using sequential index
    ax1.plot(input_seq_idx, input_close, label='Input (Historical)', color='gray', linewidth=1.5, alpha=0.7)
    ax1.plot(pred_seq_idx, gt_close, label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(pred_seq_idx, pred_close, label='Prediction', color='red', linewidth=1.5, linestyle='--')
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Prediction vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Add vertical line to separate input from prediction
    ax1.axvline(x=pred_seq_idx[0], color='green', linestyle=':', linewidth=2, alpha=0.5, label='Prediction Start')
    
    # Plot volumes using sequential index
    ax2.plot(input_seq_idx, input_volume, label='Input (Historical)', color='gray', linewidth=1.5, alpha=0.7)
    ax2.plot(pred_seq_idx, gt_volume, label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(pred_seq_idx, pred_volume, label='Prediction', color='red', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.set_xlabel('Time Step (Market Hours Only)', fontsize=14)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Volume Prediction vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Add vertical line to separate input from prediction
    ax2.axvline(x=pred_seq_idx[0], color='green', linestyle=':', linewidth=2, alpha=0.5)
    
    # Custom x-axis labels: show timestamps only at selected positions
    # Show labels at regular intervals and at key points (start, prediction start, end)
    num_ticks = min(10, len(all_timestamps))
    tick_positions = []
    tick_labels = []
    
    # Always show first and last
    tick_positions.append(0)
    tick_labels.append(all_timestamps.iloc[0].strftime('%Y-%m-%d\n%H:%M'))
    
    # Show prediction start
    pred_start_pos = len(input_df)
    if pred_start_pos not in tick_positions:
        tick_positions.append(pred_start_pos)
        tick_labels.append(all_timestamps.iloc[pred_start_pos].strftime('%Y-%m-%d\n%H:%M'))
    
    # Show last
    last_pos = len(all_timestamps) - 1
    if last_pos not in tick_positions:
        tick_positions.append(last_pos)
        tick_labels.append(all_timestamps.iloc[last_pos].strftime('%Y-%m-%d\n%H:%M'))
    
    # Add evenly spaced intermediate ticks
    step = max(1, (len(all_timestamps) - 1) // (num_ticks - len(tick_positions)))
    for i in range(step, len(all_timestamps) - 1, step):
        if i not in tick_positions:
            tick_positions.append(i)
            tick_labels.append(all_timestamps.iloc[i].strftime('%Y-%m-%d\n%H:%M'))
    
    # Sort and apply
    tick_positions, tick_labels = zip(*sorted(zip(tick_positions, tick_labels)))
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_path}")
    print(f"   Note: Market closed periods have been removed from the plot")
    # Uncomment the line below if you want to display the plot interactively
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run Kronos prediction on CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default data file (works backwards from end)
  python prediction_example.py
  
  # Specify custom CSV file
  python prediction_example.py --csv data/AAPL_5min.csv
  
  # Custom parameters
  python prediction_example.py --csv data/AAPL_5min.csv --lookback 288 --pred-len 78 --device cuda:0
  
  # Specify start date for lookback window
  python prediction_example.py --csv data/AAPL_5min.csv --start-date "2025-12-10 09:30:00" --lookback 288 --pred-len 78
  
  # Daily data with start date
  python prediction_example.py --csv data/AAPL_1d.csv --start-date "2025-12-15" --lookback 100 --pred-len 20
        """
    )
    
    parser.add_argument(
        '--csv', '-f',
        type=str,
        default="./data/XSHG_5min_600977.csv",
        help="Path to input CSV file (default: ./data/XSHG_5min_600977.csv)"
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=400,
        help="Number of historical data points to use (default: 400)"
    )
    parser.add_argument(
        '--pred-len',
        type=int,
        default=120,
        help="Number of future steps to predict (default: 120)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="Device to use: 'cuda:0', 'cpu', etc. (default: cuda:0)"
    )
    parser.add_argument(
        '--max-context',
        type=int,
        default=512,
        help="Maximum context window size (default: 512)"
    )
    parser.add_argument(
        '--temperature', '-T',
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0)"
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)"
    )
    parser.add_argument(
        '--sample-count',
        type=int,
        default=1,
        help="Number of forecast paths to generate and average (default: 1)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output plot file path (default: auto-generated with symbol and start date, e.g., prediction_result_aapl_12_11_start.png)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="NeoQuasar/Kronos-small",
        help="Model name from Hugging Face (default: NeoQuasar/Kronos-small)"
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer name from Hugging Face (default: NeoQuasar/Kronos-Tokenizer-base)"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start date for lookback window (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). If not provided, works backwards from end of CSV."
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        print(f"   Please check the path and try again.")
        sys.exit(1)
    
    print(f"📁 Loading data from: {csv_path}")
    print(f"⚙️  Parameters: lookback={args.lookback}, pred_len={args.pred_len}, device={args.device}")
    print()
    
    # 1. Load Model and Tokenizer
    print("📥 Loading model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    
    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=args.max_context)
    
    # 3. Prepare Data
    print(f"📊 Reading CSV file...")
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Validate required columns
    required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        print(f"   Required columns: {required_cols}")
        sys.exit(1)
    
    # Sort by timestamps to ensure chronological order
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    # Determine the start index based on --start-date or work backwards from end
    if args.start_date is not None:
        # Find the closest timestamp to the specified start date
        start_date_ts = pd.to_datetime(args.start_date)
        print(f"🔍 Looking for start date: {start_date_ts}")
        
        # Find the index of the closest timestamp (before or at the specified date)
        time_diffs = (df['timestamps'] - start_date_ts).abs()
        closest_idx = time_diffs.idxmin()
        closest_timestamp = df.loc[closest_idx, 'timestamps']
        time_diff = (closest_timestamp - start_date_ts).total_seconds() / 3600  # hours
        
        print(f"   Found closest timestamp: {closest_timestamp} (diff: {time_diff:.2f} hours)")
        
        # Use this as the start of the lookback window
        input_start_idx = closest_idx
        
        # Calculate prediction period indices
        pred_start_idx = input_start_idx + args.lookback
        end_idx = pred_start_idx + args.pred_len - 1
        
        # Validate we have enough data
        if end_idx >= len(df):
            print(f"⚠️  Warning: Requested period extends beyond available data")
            print(f"   Start index: {input_start_idx}, End index needed: {end_idx}, Available: {len(df) - 1}")
            if pred_start_idx >= len(df):
                print(f"❌ Error: Not enough data for lookback period. Need at least {args.lookback} rows from start date")
                sys.exit(1)
            # Adjust: use what we have
            end_idx = len(df) - 1
            actual_pred_len = end_idx - pred_start_idx + 1
            actual_lookback = args.lookback
            print(f"   Adjusted: pred_len={actual_pred_len} (requested: {args.pred_len})")
        else:
            actual_lookback = args.lookback
            actual_pred_len = args.pred_len
        
        print(f"📅 Timeline (starting from specified date):")
        print(f"   Input period: rows {input_start_idx} to {pred_start_idx-1} ({actual_lookback} data points)")
        print(f"   Prediction period: rows {pred_start_idx} to {end_idx} ({actual_pred_len} data points)")
        print(f"   Input dates: {df.loc[input_start_idx, 'timestamps']} to {df.loc[pred_start_idx-1, 'timestamps']}")
        print(f"   Ground truth dates: {df.loc[pred_start_idx, 'timestamps']} to {df.loc[end_idx, 'timestamps']}")
    else:
        # Work backwards from the end: use last x days as input, predict next y days (which exist in CSV)
        # Timeline: [start_idx ... end_idx-x-y ... end_idx-x ... end_idx]
        #           [          input (x days)          ] [ground truth (y days)]
        total_needed = args.lookback + args.pred_len
        
        if len(df) < total_needed:
            print(f"⚠️  Warning: Data has {len(df)} rows, but need at least {total_needed} rows")
            print(f"   (lookback={args.lookback} + pred_len={args.pred_len})")
            # Adjust: use what we have
            if len(df) < args.lookback:
                print(f"❌ Error: Not enough data even for lookback. Need at least {args.lookback} rows")
                sys.exit(1)
            # Use all available data for lookback, and remaining for prediction
            actual_lookback = len(df) - args.pred_len if len(df) >= args.pred_len else len(df)
            actual_pred_len = len(df) - actual_lookback
            print(f"   Adjusted: lookback={actual_lookback}, pred_len={actual_pred_len}")
        else:
            actual_lookback = args.lookback
            actual_pred_len = args.pred_len
        
        # Calculate indices: work backwards from the end
        end_idx = len(df) - 1  # Last row index
        pred_start_idx = end_idx - actual_pred_len + 1  # Start of prediction period (ground truth)
        input_start_idx = pred_start_idx - actual_lookback  # Start of input period
        
        print(f"📅 Timeline (working backwards from end):")
        print(f"   Input period: rows {input_start_idx} to {pred_start_idx-1} ({actual_lookback} data points)")
        print(f"   Prediction period: rows {pred_start_idx} to {end_idx} ({actual_pred_len} data points)")
        print(f"   Input dates: {df.loc[input_start_idx, 'timestamps']} to {df.loc[pred_start_idx-1, 'timestamps']}")
        print(f"   Ground truth dates: {df.loc[pred_start_idx, 'timestamps']} to {df.loc[end_idx, 'timestamps']}")
    
    # Extract input data (x days before prediction period)
    x_df = df.loc[input_start_idx:pred_start_idx-1, ['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
    x_timestamp = df.loc[input_start_idx:pred_start_idx-1, 'timestamps'].reset_index(drop=True)
    
    # Extract ground truth timestamps for prediction period (y days)
    y_timestamp = df.loc[pred_start_idx:end_idx, 'timestamps'].reset_index(drop=True)
    
    # For plotting: show input period + prediction period (with ground truth for comparison)
    kline_df = df.loc[input_start_idx:end_idx].copy()
    
    # 4. Make Prediction
    print(f"🔮 Making predictions...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True
    )
    
    # 5. Visualize Results
    print("\n📈 Forecasted Data Head:")
    print(pred_df.head())
    
    # kline_df is already set correctly on line 204: df.loc[input_start_idx:end_idx]
    # It contains: input period (actual_lookback rows) + prediction period (actual_pred_len rows)
    # For plotting, we need to know how many rows are input vs prediction
    input_len = actual_lookback  # First part of kline_df is input
    
    # Generate output filename if not provided
    if args.output is None:
        # Extract symbol from CSV filename (e.g., "AAPL_5min.csv" -> "AAPL")
        csv_stem = csv_path.stem  # Get filename without extension
        # Try to extract symbol (assumes format like "AAPL_5min" or "AAPL" or "XSHG_5min_600977")
        symbol = None
        if '_' in csv_stem:
            # Try first part as symbol
            potential_symbol = csv_stem.split('_')[0].upper()
            # Check if it looks like a stock symbol (letters only, 1-5 chars)
            if potential_symbol.isalpha() and 1 <= len(potential_symbol) <= 5:
                symbol = potential_symbol.lower()
        else:
            # Whole filename might be the symbol
            if csv_stem.isalpha() and 1 <= len(csv_stem) <= 5:
                symbol = csv_stem.lower()
        
        # Fallback: use "unknown" if we can't extract symbol
        if symbol is None:
            symbol = "unknown"
        
        # Get start date from actual data
        start_timestamp = df.loc[input_start_idx, 'timestamps']
        # Format as MM_DD (e.g., 12_11 for December 11)
        date_str = f"{start_timestamp.month:02d}_{start_timestamp.day:02d}"
        
        # Generate filename: prediction_result_aapl_12_11_start.png
        output_dir = csv_path.parent / "data" if csv_path.parent.name != "data" else csv_path.parent
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f"prediction_result_{symbol}_{date_str}_start.png"
    else:
        plot_path = Path(args.output)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_prediction(kline_df, pred_df, input_len, plot_path=plot_path)
    
    print(f"\n✅ Prediction complete! Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()


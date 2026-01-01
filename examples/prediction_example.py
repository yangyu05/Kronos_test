import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, plot_path="prediction_result.png"):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    # Save plot instead of showing (useful for headless servers)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_path}")
    # Uncomment the line below if you want to display the plot interactively
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run Kronos prediction on CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default data file
  python prediction_example.py
  
  # Specify custom CSV file
  python prediction_example.py --csv data/AAPL_1d.csv
  
  # Custom parameters
  python prediction_example.py --csv data/AAPL_1d.csv --lookback 500 --pred-len 100 --device cpu
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
        default="prediction_result.png",
        help="Output plot file path (default: prediction_result.png)"
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
    
    # Check if we have enough data
    if len(df) < args.lookback + args.pred_len:
        print(f"⚠️  Warning: Data has {len(df)} rows, but need at least {args.lookback + args.pred_len} rows")
        print(f"   Using available data: lookback={min(args.lookback, len(df) - args.pred_len)}, pred_len={args.pred_len}")
        actual_lookback = min(args.lookback, len(df) - args.pred_len)
        if actual_lookback <= 0:
            print(f"❌ Error: Not enough data for prediction")
            sys.exit(1)
    else:
        actual_lookback = args.lookback
    
    x_df = df.loc[:actual_lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:actual_lookback-1, 'timestamps']
    
    # For y_timestamp, use actual future timestamps if available, otherwise generate
    if len(df) >= actual_lookback + args.pred_len:
        y_timestamp = df.loc[actual_lookback:actual_lookback+args.pred_len-1, 'timestamps']
    else:
        # Generate timestamps based on the last timestamp and frequency
        last_ts = df.loc[actual_lookback-1, 'timestamps']
        freq = pd.infer_freq(df['timestamps'].head(10))
        if freq is None:
            # Default to daily if can't infer
            freq = 'D'
        y_timestamp = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=args.pred_len, freq=freq)
        print(f"   Generated {args.pred_len} future timestamps (freq={freq})")
    
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
    
    # Combine historical and forecasted data for plotting
    if len(df) >= actual_lookback + args.pred_len:
        kline_df = df.loc[:actual_lookback+args.pred_len-1]
    else:
        kline_df = df.loc[:actual_lookback]
    
    # Update plot function to use custom output path
    plot_path = args.output
    plot_prediction(kline_df, pred_df, plot_path=plot_path)
    
    print(f"\n✅ Prediction complete! Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()


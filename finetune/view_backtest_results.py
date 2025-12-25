"""
Script to view and analyze backtest results.
"""
import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append("../")

def view_backtest_results():
    """View saved backtest predictions and generate summary."""
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to predictions file
    results_dir = os.path.join(project_root, "outputs/backtest_results/finetune_backtest_us_market")
    predictions_file = os.path.join(results_dir, "predictions.pkl")
    
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        print("\nPlease run the backtest first:")
        print("  python finetune/qlib_test.py --device cuda:0 --symbols AAPL MSFT GOOGL")
        return
    
    print(f"Loading predictions from: {predictions_file}\n")
    
    # Load predictions
    with open(predictions_file, 'rb') as f:
        model_preds = pickle.load(f)
    
    print("="*70)
    print("BACKTEST PREDICTIONS SUMMARY")
    print("="*70)
    
    # Display summary for each signal type
    for signal_name, pred_df in model_preds.items():
        print(f"\n📊 Signal Type: {signal_name}")
        print("-" * 70)
        print(f"Shape: {pred_df.shape} (dates x symbols)")
        print(f"Date range: {pred_df.index.min()} to {pred_df.index.max()}")
        print(f"Symbols: {list(pred_df.columns)}")
        
        # Show statistics
        print(f"\nStatistics:")
        print(pred_df.describe())
        
        # Show first few rows
        print(f"\nFirst 10 dates:")
        print(pred_df.head(10))
        
        print("\n" + "="*70)
    
    # Option to plot
    print("\n💡 Tip: To visualize the results, check the console output from the backtest run")
    print("   or check if a plot was saved to: ../figures/backtest_result_example.png")
    
    # Try to load and display plot if it exists
    plot_path = os.path.join(project_root, "figures/backtest_result_example.png")
    if os.path.exists(plot_path):
        print(f"\n✅ Plot found at: {plot_path}")
        print("   You can view it with: xdg-open figures/backtest_result_example.png")
    else:
        print(f"\n⚠️  Plot not found at: {plot_path}")
        print("   The plot should have been generated during the backtest run")

if __name__ == '__main__':
    view_backtest_results()


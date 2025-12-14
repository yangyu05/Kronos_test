#!/usr/bin/env python3
"""
Script to download and prepare US market data for Qlib.

This script downloads US market data using Qlib's built-in data downloader
and prepares it for use with the Kronos model.

Usage:
    python finetune/prepare_us_data.py
"""

import os
import subprocess
import sys
from pathlib import Path


def download_qlib_us_data(target_dir: str = "~/.qlib/qlib_data/us_data"):
    """
    Download US market data using Qlib's CLI.
    
    Args:
        target_dir: Directory where the US market data will be stored.
    """
    # Expand user path
    target_dir = os.path.expanduser(target_dir)
    
    print(f"Downloading US market data to: {target_dir}")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Use qlib CLI to download US market data
        cmd = [
            sys.executable,
            "-m",
            "qlib.cli.data",
            "qlib_data",
            "--target_dir",
            target_dir,
            "--region",
            "us"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Data download completed successfully!")
        print(f"Data stored at: {target_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def verify_data_download(target_dir: str = "~/.qlib/qlib_data/us_data"):
    """
    Verify that the data was downloaded successfully.
    
    Args:
        target_dir: Directory where the US market data should be stored.
    """
    target_dir = os.path.expanduser(target_dir)
    
    if not os.path.exists(target_dir):
        print(f"ERROR: Data directory does not exist: {target_dir}")
        return False
    
    # Check for key qlib data files
    required_files = ["calendars", "instruments"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(target_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"WARNING: Some required files are missing: {missing_files}")
        return False
    
    print("Data verification passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Qlib US Market Data Preparation Script")
    print("=" * 60)
    print()
    
    # Default data directory
    data_dir = "~/.qlib/qlib_data/us_data"
    
    # Check if data already exists
    expanded_dir = os.path.expanduser(data_dir)
    if os.path.exists(expanded_dir) and verify_data_download(data_dir):
        print(f"\nUS market data already exists at: {expanded_dir}")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            sys.exit(0)
    
    # Download data
    success = download_qlib_us_data(data_dir)
    
    if success:
        # Verify download
        if verify_data_download(data_dir):
            print("\n" + "=" * 60)
            print("US market data preparation completed successfully!")
            print("=" * 60)
            print(f"\nNext steps:")
            print(f"1. Update finetune/config.py to use US market settings")
            print(f"2. Run finetune/qlib_data_preprocess.py to process the data")
        else:
            print("\nWARNING: Data download completed but verification failed.")
            print("Please check the data directory manually.")
            sys.exit(1)
    else:
        print("\nERROR: Failed to download US market data.")
        print("Please check the error messages above and try again.")
        sys.exit(1)


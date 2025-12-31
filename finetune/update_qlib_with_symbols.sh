#!/bin/bash
# Helper script to update Qlib data with specific symbols

if [ $# -eq 0 ]; then
    echo "Usage: $0 SYMBOL1 SYMBOL2 ..."
    echo "Example: $0 AAPL MSFT GOOGL AMZN TSLA"
    exit 1
fi

# Join symbols with comma
SYMBOLS=$(IFS=,; echo "$*")

echo "Updating Qlib data for symbols: $SYMBOLS"
echo "Date range: 2020-11-11 to $(date +%Y-%m-%d)"
echo ""

cd ~/qlib_repo/scripts/data_collector/yahoo

QLIB_CUSTOM_SYMBOLS="$SYMBOLS" python3 collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/us_data \
    --trading_date 2020-11-11 \
    --end_date $(date +%Y-%m-%d) \
    --region US \
    --interval 1d


import backtrader as bt
import pandas as pd
from datetime import datetime
import json
import os

# Import our data fetcher and strategy
from crew_agents.src.data_fetcher import fetch_ohlcv
from strategies.sma_crossover import SmaCrossOverStrategy

def run_optimization():
    # --- Configuration ---
    symbol = 'BTC/USD'  # Trading pair
    timeframe = '1h'         # Timeframe for data
    start_date = '2024-01-01T00:00:00Z' # Start date for historical data
    initial_cash = 10000.0   # Starting portfolio value
    commission_rate = 0.001  # Commission rate (e.g., 0.1%)
    limit = 2000

    # --- Fetch Data ---
    print(f"Fetching {symbol} {timeframe} data...")
    data_df = fetch_ohlcv(symbol, timeframe, since=start_date, limit=limit)
    if data_df is None or data_df.empty:
        print(f"Could not fetch data for {symbol}.")
        return
    
    print(f"Successfully fetched {len(data_df)} data points for {symbol}.")

    # Convert DataFrame to Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None, # Use the DataFrame index for datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1 # Not available
    )

    # --- Backtrader Setup ---
    cerebro = bt.Cerebro()

    # Add strategy - Use optstrategy for optimization
    strats = cerebro.optstrategy(
        SmaCrossOverStrategy,
        pfast=range(5, 21, 1),  # Fast MA range: 5 to 20
        pslow=range(20, 61, 5)  # Slow MA range: 20 to 60, step 5
    )

    # Add data feed
    cerebro.adddata(data)

    # Set starting cash
    cerebro.broker.setcash(initial_cash)

    # Set commission
    cerebro.broker.setcommission(commission=commission_rate)

    # Add a sizer for position sizing (e.g., 95% of portfolio)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Add analyzers (optional, but useful)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print(f'Starting Portfolio Value: {initial_cash:.2f}')

    # Run Optimization
    results = cerebro.run(maxcpus=1) # Use maxcpus=1 for simplicity first

    # --- Process Optimization Results ---
    print("--- Processing optimization results ---")
    parsed_results = []
    for run_results in results:
        # run_results[0] holds OptReturn (parameters and finalized analyzers)
        if not run_results: # Basic check if the list itself is empty
            print("Skipping empty run_results list.")
            continue
            
        try:
            # Access OptReturn object directly
            opt_return_obj = run_results[0]

            # Get parameters from OptReturn object's params attribute
            try:
                pfast = opt_return_obj.params.pfast
                pslow = opt_return_obj.params.pslow
            except AttributeError as e:
                print(f"Error accessing parameters from OptReturn.params: {e} for run: {run_results}")
                continue # Skip if parameters cannot be retrieved

            print(f"\nProcessing run for pfast={pfast}, pslow={pslow}")

            # --- Get analyzer results directly from OptReturn object --- 
            total_return = None
            final_value = initial_cash # Default if calculation fails
            
            # (Using try-except blocks for safety)
            try:
                sharpe_analysis = opt_return_obj.analyzers.sharpe_ratio.get_analysis()
                sharpe = sharpe_analysis.get('sharperatio', None) if sharpe_analysis else None
                print(f"  Sharpe Analysis: {sharpe_analysis}, Sharpe: {sharpe}")
            except AttributeError:
                print("  Sharpe Ratio analyzer not found on OptReturn.")
                sharpe = None

            try:
                drawdown_analysis = opt_return_obj.analyzers.drawdown.get_analysis()
                drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0) if drawdown_analysis else 0
                print(f"  Drawdown Analysis: {drawdown_analysis}, Drawdown: {drawdown}")
            except AttributeError:
                print("  Drawdown analyzer not found on OptReturn.")
                drawdown = 0

            try:
                returns_analysis = opt_return_obj.analyzers.returns.get_analysis()
                total_return = returns_analysis.get('rtot', None) if returns_analysis else None # Total return as a fraction (e.g., 0.1 for 10%)
                print(f"  Returns Analysis: {returns_analysis}, Total Return: {total_return}")
                if total_return is not None:
                     final_value = initial_cash * (1 + total_return)
                     print(f"  Calculated Final Value from Returns: {final_value}")
                else:
                    print("  Total return ('rtot') not found in Returns analyzer.")
            except AttributeError:
                print("  Returns analyzer not found on OptReturn.")
                total_return = None # Ensure it's None if analyzer missing
                print(f"  Could not calculate Final Value from Returns.")
            
            try:
                trade_analyzer_analysis = opt_return_obj.analyzers.trade_analyzer.get_analysis()
                trades = trade_analyzer_analysis.get('total', {}).get('total', 0) if trade_analyzer_analysis else 0
                print(f"  Trade Analysis: {trade_analyzer_analysis}, Trades: {trades}")
            except AttributeError:
                print("  Trade analyzer not found on OptReturn.")
                trades = 0

            # Append results (even if some metrics are None/default)
            parsed_results.append({
                'pfast': pfast,
                'pslow': pslow,
                'final_value': final_value, # Use calculated value
                'sharpe': sharpe,
                'max_drawdown': drawdown,
                'total_return': total_return, # Store the fractional return
                'trades': trades
            })
            print(f"  Appended results for pfast={pfast}, pslow={pslow}")
            
        except (AttributeError, IndexError) as e:
             print(f"Error processing run results: {e} for run: {run_results}")
             continue # Skip this run if accessing elements failed

    # Sort results
    print(f"\nTotal parsed results before sorting: {len(parsed_results)}") 
    sorted_results = sorted(parsed_results, key=lambda x: x['final_value'], reverse=True)

    # Add diagnostic print just before saving
    print(f"\nAttempting to save {len(sorted_results)} results to JSON.")
    if sorted_results:
        print(f"First result item: {sorted_results[0]}")
    else:
        print("sorted_results list is empty before saving!")

    # Save results to JSON file
    results_filepath = 'optimization_results.json'
    print(f"\n--- Saving optimization results to {results_filepath} --- ")
    try:
        with open(results_filepath, 'w') as f:
            print(f"Length of sorted_results just before json.dump: {len(sorted_results)}")
            json.dump(sorted_results, f, indent=4)
            f.flush()  
            os.fsync(f.fileno()) 
        print(f"Successfully saved results to {results_filepath}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

if __name__ == '__main__':
    run_optimization()

def run_backtest():
    print("This function is deprecated. Run run_optimization() instead.")

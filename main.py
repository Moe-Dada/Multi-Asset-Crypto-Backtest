import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import backtrader as bt
import warnings
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing as mp

# Suppress warnings
warnings.filterwarnings('ignore')

# List of assets to trade
ASSETS = ['BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ETH/USDT', 'DOGE/USDT']
BENCHMARK = 'BTC/USDT'

# Fetch historical data using ccxt
def get_ccxt_data(symbol, timeframe, from_date, to_date):
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{from_date}T00:00:00Z')
    to_timestamp = exchange.parse8601(f'{to_date}T00:00:00Z')
    all_ohlcv = []
    limit = 1000
    while since < to_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < limit:
                break
        except ccxt.NetworkError as e:
            print(f"Network error while fetching {symbol}: {e}")
            break
        except ccxt.ExchangeError as e:
            print(f"Exchange error while fetching {symbol}: {e}")
            break
    df = pd.DataFrame(all_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

# Define the date range for data fetching
from_date = '2020-01-01'
to_date = datetime.utcnow().strftime('%Y-%m-%d')

# Initialize a dictionary to store dataframes for each asset
data_frames = {}

# Fetch data for all assets
for asset in ASSETS + [BENCHMARK]:
    print(f"Fetching data for {asset}...")
    df = get_ccxt_data(asset, '1d', from_date, to_date)
    if df.empty:
        print(f"No data fetched for {asset}. Please check the symbol or try a different date range.")
    else:
        data_frames[asset] = df

# Check if all assets have been fetched successfully
if len(data_frames) != len(ASSETS) + 1:
    missing_assets = set(ASSETS + [BENCHMARK]) - set(data_frames.keys())
    print(f"Missing data for assets: {missing_assets}. Exiting the script.")
    exit()

# Find the common date index across all assets
common_index = data_frames[ASSETS[0]].index
for asset in ASSETS[1:] + [BENCHMARK]:
    common_index = common_index.intersection(data_frames[asset].index)

# Trim all dataframes to the common index
for asset in ASSETS + [BENCHMARK]:
    data_frames[asset] = data_frames[asset].loc[common_index]

# Handle missing data by forward and backward filling
for asset in ASSETS + [BENCHMARK]:
    if data_frames[asset].isnull().values.any():
        print(f"Filling missing data for {asset}...")
        data_frames[asset].fillna(method='ffill', inplace=True)
        data_frames[asset].fillna(method='bfill', inplace=True)

# Verify data alignment
for asset in ASSETS + [BENCHMARK]:
    assert len(data_frames[asset]) == len(common_index), f"Data length mismatch for {asset}"

# Combine assets' data into separate dataframes
asset_data = {}
for asset in ASSETS:
    asset_df = data_frames[asset][['open', 'high', 'low', 'close', 'volume']].copy()
    asset_df.columns = [f"{asset.split('/')[0].lower()}_{col}" for col in asset_df.columns]  # e.g., bnb_close
    asset_data[asset] = asset_df

# Define a Custom DrawDown Analyzer (Optional)
class CustomDrawDown(bt.Analyzer):
    def __init__(self):
        self.highest = -float('inf')
        self.max_drawdown = 0.0
        self.drawdowns = []

    def start(self):
        self.highest = self.strategy.broker.getvalue()

    def next(self):
        current_value = self.strategy.broker.getvalue()
        if current_value > self.highest:
            self.highest = current_value
        drawdown = (self.highest - current_value) / self.highest * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.drawdowns.append(drawdown)

    def get_analysis(self):
        return {'max_drawdown': self.max_drawdown, 'drawdowns': self.drawdowns}

# Define the strategy
class MultiAssetStrategy(bt.Strategy):
    params = (
        ('sma_long_period', 50),
        ('sma_short_period', 20),
        ('stop_loss_pct', 1.0),  # Stop-loss percentage
        ('position_size', 0.1),  # 10% of portfolio
    )

    def __init__(self):
        # Reference to BTC data (benchmark)
        self.btc_close = self.datas[0].close

        # Calculate BTC SMAs
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.btc_close, period=self.params.sma_long_period
        )
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.btc_close, period=self.params.sma_short_period
        )

        # Initialize dictionaries to track asset positions, entry prices, and stop-loss orders
        self.asset_positions = {asset: None for asset in ASSETS}
        self.entry_prices = {asset: None for asset in ASSETS}
        self.stop_orders = {asset: None for asset in ASSETS}

        # Record portfolio values for manual drawdown calculation
        self.portfolio_values = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                print(f"Bought {order.data._name} at {order.executed.price:.2f}")
                # Record entry price
                self.entry_prices[order.data._name] = order.executed.price
                # Calculate stop-loss price for long position
                stop_price = order.executed.price * (1 - self.params.stop_loss_pct / 100)
                # Place stop-loss order WITHOUT parent=order
                stop_order = self.sell(
                    data=order.data,
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    size=order.size
                )
                self.stop_orders[order.data._name] = stop_order
                print(f"Placed Stop-Loss for {order.data._name} at {stop_price:.2f}")

            elif order.issell():
                if order.size > 0:
                    print(f"Shorted {order.data._name} at {order.executed.price:.2f}")
                    # Record entry price for short position
                    self.entry_prices[order.data._name] = order.executed.price
                    # Calculate stop-loss price for short position
                    stop_price = order.executed.price * (1 + self.params.stop_loss_pct / 100)
                    # Place stop-loss order WITHOUT parent=order
                    stop_order = self.buy(
                        data=order.data,
                        exectype=bt.Order.Stop,
                        price=stop_price,
                        size=order.size
                    )
                    self.stop_orders[order.data._name] = stop_order
                    print(f"Placed Stop-Loss for {order.data._name} at {stop_price:.2f}")
                else:
                    print(f"Covered {order.data._name} at {order.executed.price:.2f}")
            print(f"Updated Portfolio Value: {self.broker.getvalue():.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order {order.getstatusname()} for {order.data._name}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        print(f"Trade closed: {trade.data._name}, PnL: {trade.pnl:.2f}")
        # Reset entry price and stop order
        self.entry_prices[trade.data._name] = None
        self.stop_orders[trade.data._name] = None

    def next(self):
        # Get the latest BTC close and SMAs
        btc_close = self.btc_close[0]
        sma_long = self.sma_long[0]
        sma_short = self.sma_short[0]

        # Define conditions
        long_condition = btc_close > sma_long and btc_close > sma_short
        short_condition = btc_close < sma_long and btc_close < sma_short

        for asset in ASSETS:
            asset_data = self.getdatabyname(asset.split('/')[0])
            asset_close = asset_data.close[0]
            position = self.getposition(asset_data).size

            if long_condition:
                if position is None or position <= 0:
                    # Close short position if exists
                    if position < 0:
                        self.close(data=asset_data)
                    # Allocate position_size percentage of the portfolio to each asset for long positions
                    allocation = self.params.position_size * self.broker.getvalue()
                    size = allocation / asset_close
                    self.buy(data=asset_data, size=size)
                    self.asset_positions[asset] = 'long'

            elif short_condition:
                if position is None or position >= 0:
                    # Close long position if exists
                    if position > 0:
                        self.close(data=asset_data)
                    # Allocate position_size percentage of the portfolio to each asset for short positions
                    allocation = self.params.position_size * self.broker.getvalue()
                    size = allocation / asset_close
                    self.sell(data=asset_data, size=size)  # Enter short
                    self.asset_positions[asset] = 'short'

            else:
                # Exit long position if price falls below the short SMA
                if position > 0 and btc_close < sma_short:
                    self.close(data=asset_data)
                    self.asset_positions[asset] = None

                # Exit short position if price rises above the short SMA
                if position < 0 and btc_close > sma_short:
                    self.close(data=asset_data)
                    self.asset_positions[asset] = None

            # Cancel existing stop-loss orders if position is closed manually
            if (position == 0 or (position is None)) and self.stop_orders[asset]:
                self.cancel(self.stop_orders[asset])
                self.stop_orders[asset] = None
                self.entry_prices[asset] = None

        # Record the current portfolio value
        self.portfolio_values.append(self.broker.getvalue())

    def stop(self):
        # Ensure all positions are closed at the end
        for data in self.datas:
            if self.getposition(data).size:
                self.close(data=data)
                print(f"Final position closed for {data._name} at {data.close[0]:.2f}")

# Function to run backtest with given parameters
def run_backtest(params, data_frames, ASSETS, BENCHMARK):
    cerebro = bt.Cerebro()

    # Add analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name='sharpe_ratio',
        timeframe=bt.TimeFrame.Days,
        compression=1,
        riskfreerate=0.0,
        annualize=True
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        timeframe=bt.TimeFrame.Days,
        compression=1,
        _name='time_return'
    )
    cerebro.addanalyzer(CustomDrawDown, _name='custom_drawdown')

    # Add BTC data as the primary data feed
    btc_data_feed = bt.feeds.PandasData(
        dataname=data_frames[BENCHMARK],
        name='BTC'
    )
    cerebro.adddata(btc_data_feed)

    # Add each asset's data feed
    for asset in ASSETS:
        data_feed = bt.feeds.PandasData(
            dataname=data_frames[asset],
            name=asset.split('/')[0]
        )
        cerebro.adddata(data_feed)

    # Add the strategy with current parameters
    cerebro.addstrategy(MultiAssetStrategy, **params)

    # Set initial cash and commission
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)

    # Run the backtest with analyzers
    results = cerebro.run()
    strat = results[0]

    # Collect analyzer results
    # Sharpe Ratio
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
    sharpe_ratio_value = sharpe_ratio.get('sharperatio', None)

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown.get('max', {}).get('drawdown', None)

    # Trade Analyzer
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analyzer.get('total', {}).get('total', 0)
    won_trades = trade_analyzer.get('won', {}).get('total', 0)
    lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else None

    # Time Return
    time_return = strat.analyzers.time_return.get_analysis()
    cumulative_return = (pd.Series(time_return).add(1).prod() - 1) * 100  # In percentage

    # Custom Drawdown
    custom_drawdown = strat.analyzers.custom_drawdown.get_analysis()
    max_custom_drawdown = custom_drawdown.get('max_drawdown', None)

    # Collect all scalar results into 'analyzers' dict
    analyzers = {
        'sharpe_ratio': sharpe_ratio_value,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'won_trades': won_trades,
        'lost_trades': lost_trades,
        'win_rate': win_rate,
        'cumulative_return': cumulative_return,
        'max_custom_drawdown': max_custom_drawdown,
        'final_portfolio_value': cerebro.broker.getvalue(),
    }
    # Add parameters to 'analyzers' dict
    analyzers.update(params)

    return analyzers

# Define parameter grid
sma_long_periods = [30, 50, 100]
sma_short_periods = [10, 20, 30]
stop_loss_pcts = [0.5, 1.0, 1.5]
position_sizes = [0.05, 0.1, 0.2]

param_grid = list(product(sma_long_periods, sma_short_periods, stop_loss_pcts, position_sizes))

# Convert to list of dictionaries
param_combinations = []
for sma_long, sma_short, stop_loss, position_size in param_grid:
    # Ensure sma_short < sma_long to make logical sense
    if sma_short < sma_long:
        param_combinations.append({
            'sma_long_period': sma_long,
            'sma_short_period': sma_short,
            'stop_loss_pct': stop_loss,
            'position_size': position_size
        })

print(f"Total parameter combinations to backtest: {len(param_combinations)}")

# Function to execute backtests in parallel
def parallel_backtest(params):
    return run_backtest(params, data_frames, ASSETS, BENCHMARK)

if __name__ == '__main__':
    # Determine the number of processes to use
    num_processes = min(mp.cpu_count(), len(param_combinations))
    print(f"Running backtests on {num_processes} processes...")

    # Run backtests
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(parallel_backtest, param_combinations)

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Sort and display top strategies by Sharpe Ratio
    top_strategies = results_df.sort_values(by='sharpe_ratio', ascending=False).head(5)
    print("\nTop 5 Strategies by Sharpe Ratio:")
    print(top_strategies[['sma_long_period', 'sma_short_period', 'stop_loss_pct', 'position_size',
                          'sharpe_ratio', 'max_drawdown', 'win_rate', 'cumulative_return', 'final_portfolio_value']])

    # Save all results to a CSV for further analysis
    results_df.to_csv('backtest_results.csv', index=False)
    print("\nAll backtest results have been saved to 'backtest_results.csv'.")


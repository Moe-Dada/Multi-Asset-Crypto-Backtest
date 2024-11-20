# Multi-Asset Cryptocurrency Trading Backtest

![License](https://img.shields.io/github/license/Moe-Dada/Multi-Asset-Crypto-Backtest)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Backtrader](https://img.shields.io/badge/Backtrader-1.9.78.123-brightgreen)
![CCXT](https://img.shields.io/badge/CCXT-1.95.72-yellow)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Analyzers](#analyzers)
- [Contact](#contact)
- [FAQs](#faqs)
- [Troubleshooting](#troubleshooting)

## Overview

The **Multi-Asset Cryptocurrency Trading Backtest** is a comprehensive Python script designed to backtest a multi-asset trading strategy using historical cryptocurrency data. Leveraging the power of the [Backtrader](https://www.backtrader.com/) framework and the [CCXT](https://github.com/ccxt/ccxt) library for fetching data, this tool allows traders and analysts to evaluate the performance of their strategies across multiple cryptocurrencies with ease.

## Features

- **Multi-Asset Support**: Trade multiple cryptocurrencies simultaneously.
- **Customizable Strategy Parameters**: Optimize moving average periods, stop-loss percentages, and position sizes.
- **Parallel Backtesting**: Utilize multiprocessing to run multiple backtests concurrently.
- **Comprehensive Analyzers**: Evaluate strategies using Sharpe Ratio, Drawdown, Trade Analysis, Time Return, and Custom Drawdown metrics.
- **Data Handling**: Fetches and preprocesses historical data from Binance using CCXT.
- **Result Export**: Saves backtest results to a CSV file for further analysis.
- **Robust Error Handling**: Manages network and exchange errors gracefully during data fetching.

## Installation

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/)

### Clone the Repository

git clone https://github.com/Moe-Dada/Multi-Asset-Crypto-Backtest.git
cd multi-asset-trading-backtest

Create a Virtual Environment (Optional but Recommended)

### Copy code
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install Dependencies

- bash
- Copy code
- pip install -r requirements.txt
- requirements.tx
Note: Ensure that you have the latest versions of the libraries for optimal performance.

### Usage
The main script performs the following steps:

- Data Fetching: Retrieves historical OHLCV data for specified cryptocurrencies from Binance.
- Data Preprocessing: Aligns data across all assets, handles missing values, and prepares data feeds for Backtrader.
- Strategy Definition: Implements a multi-asset trading strategy based on moving averages and stop-loss mechanisms.
- Backtesting: Runs backtests across a grid of strategy parameters in parallel.
- Results Compilation: Aggregates and saves backtest results for analysis.
- Running the Backtest
- Ensure you are in the project directory and your virtual environment is activated (if using one).


### Output
- Console Output: Displays progress messages, including data fetching status, backtest progress, and top-performing strategies.
- CSV File: backtest_results.csv containing detailed results of all backtested parameter combinations.
- Configuration
- Assets to Trade
- Modify the ASSETS and BENCHMARK variables to change the assets you wish to trade and the benchmark asset.

Date Range
- Set the from_date and to_date variables to define the period for which you want to fetch historical data.
from_date = '2020-01-01'
to_date = datetime.utcnow().strftime('%Y-%m-%d')

Parameter Grid
- Customize the parameter grid for the backtest by modifying the lists:
sma_long_periods = [30, 50, 100]
sma_short_periods = [10, 20, 30]
stop_loss_pcts = [0.5, 1.0, 1.5]
position_sizes = [0.05, 0.1, 0.2]

Ensure that sma_short_period < sma_long_period for logical moving average configurations.

### Code Structure

main.py (Main Script)
- Imports: Essential libraries and modules.
- Constants: Asset lists and benchmark definitions.
- Data Fetching Function: get_ccxt_data retrieves historical data from Binance.
- Data Preprocessing: Aligns and cleans data across all assets.
- Custom Analyzers: CustomDrawDown for additional drawdown metrics.
- Strategy Class: MultiAssetStrategy defines the trading logic.
- Backtest Execution: run_backtest runs the strategy with given parameters.
- Parameter Grid Definition: Creates combinations of strategy parameters.
- Parallel Execution: Utilizes multiprocessing to run backtests concurrently.
- Results Handling: Compiles and saves backtest results.

### Analyzers
The backtest incorporates several analyzers to evaluate strategy performance:

- Sharpe Ratio: Measures risk-adjusted return.
- Drawdown: Evaluates the maximum loss from a peak.
- Trade Analyzer: Provides insights into trade statistics (total trades, win rate, etc.).
- Time Return: Calculates cumulative returns over time.
- Custom DrawDown: Custom implementation for additional drawdown analysis.


### Contact
For any questions or support, please open an issue in the GitHub repository or contact mosesdadaphd@outlook.com.

### Acknowledgements
Backtrader for the robust backtesting framework.
CCXT for providing cryptocurrency market data.
The open-source community for continuous support and contributions.
Additional Resources
Backtrader Documentation
CCXT Documentation
Pandas Documentation
Multiprocessing in Python

### FAQs
- Q1: How can I add more assets to trade?

Edit the ASSETS list in the script to include additional trading pairs supported by Binance.

- Q2: Can I change the data source from Binance to another exchange?

Yes, modify the ccxt.binance() instance in the get_ccxt_data function to your preferred exchange supported by CCXT.

- Q3: How do I interpret the backtest results?

Review the backtest_results.csv file, focusing on metrics like Sharpe Ratio, Max Drawdown, Win Rate, and Cumulative Return to evaluate strategy performance.

### Troubleshooting
- Data Fetching Issues: Ensure your internet connection is stable. Verify that the asset symbols are correct and supported by Binance.
Insufficient Data: Some assets might have limited historical data. Try adjusting the from_date or selecting different assets.
- Performance Bottlenecks: Running a large number of backtests may consume significant CPU resources. Adjust the num_processes based on your system's capabilities.

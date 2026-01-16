# Stock Analysis Dashboard

A Plotly Dash web application for analyzing stock data and relative strength metrics.

## Features

- View candlestick charts with volume for individual stocks
- Analyze relative strength metrics:
  - Relative Strength
  - RS Rank (Index-specific)
  - RS Rank (Overall)
  - RS Z-Score
- Screen stocks based on:
  - RS Rank ranges
  - RS Z-Score ranges
  - Volume standard deviation thresholds

## Prerequisites

- Python 3.8+
- PostgreSQL database with the following tables:
  - dim_etf
  - dim_stock
  - junc_stock_etf
  - fact_index_ohlc
  - fact_commons_ohlc
  - fact_stock_metrics

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure database connection:
   - Open `config.py`
   - Update the database connection parameters:
     ```python
     DB_CONFIG = {
         'dbname': 'stocks',
         'user': 'your_username',
         'password': 'your_password',
         'host': 'localhost',
         'port': '5432'
     }
     ```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8050
```

3. Use the application:
   - Select a stock ticker and date range to view charts
   - Use the Stock Screener tab to filter stocks based on various metrics
   - View results in a sortable table format

## Project Structure

- `app.py`: Main application file with Dash layout and callbacks
- `database.py`: Database connection and query functions
- `config.py`: Configuration settings
- `requirements.txt`: Python package dependencies 
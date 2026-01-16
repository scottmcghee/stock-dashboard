import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from database import (
    fetch_stock_data, 
    fetch_stock_metrics, 
    search_stocks_by_metrics,
    get_all_tickers,
    get_all_etfs,
    fetch_latest_etf_metrics,
    fetch_etf_ohlcv,
    fetch_etf_metrics,
    get_all_option_symbols,
    fetch_option_last_30_days,
    fetch_option_latest_comparison,
    fetch_option_column_history,
    get_option_numerical_columns,
    fetch_option_screener_results
)
from dash import callback_context

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout components
ticker_dropdown = dbc.Select(
    id='ticker-select',
    options=[{'label': ticker, 'value': ticker} for ticker in get_all_tickers()],
    placeholder="Select a stock ticker..."
)

interval_dropdown = dbc.Select(
    id='interval-select',
    options=[
        {'label': 'Daily', 'value': 'D'},
        {'label': 'Weekly', 'value': 'W'},
        {'label': 'Monthly', 'value': 'M'}
    ],
    value='D',
    style={"width": "150px"}
)

date_range = dcc.DatePickerRange(
    id='date-range',
    start_date=(datetime.now() - timedelta(days=365)).date(),
    end_date=datetime.now().date(),
    display_format='YYYY-MM-DD'
)

# Add ETF dropdown for Single Stock Analysis
etf_dropdown = dcc.Dropdown(
    id='etf-select',
    options=[],
    placeholder="Select ETF (optional)",
    style={"width": "100%"}
)

metrics_filters = dbc.Form([
    dbc.Row([
        dbc.Col([
            html.Label("ETF Filter"),
            dbc.Select(
                id="etf-filter",
                options=[
                    {"label": "All ETFs", "value": None},
                    *[{"label": etf, "value": etf} for etf in get_all_etfs()]
                ],
                value=None
            )
        ]),
        dbc.Col([
            html.Label("RS Rank Range"),
            dbc.Row([
                dbc.Col(dbc.Input(id="rs-rank-min", type="number", placeholder="Min", min=0, max=100)),
                dbc.Col(dbc.Input(id="rs-rank-max", type="number", placeholder="Max", min=0, max=100))
            ])
        ]),
        dbc.Col([
            html.Label("RS Rank Overall Range"),
            dbc.Row([
                dbc.Col(dbc.Input(id="rs-rank-overall-min", type="number", placeholder="Min", min=0, max=100)),
                dbc.Col(dbc.Input(id="rs-rank-overall-max", type="number", placeholder="Max", min=0, max=100))
            ])
        ]),
        dbc.Col([
            html.Label("RS Z-Score Range"),
            dbc.Row([
                dbc.Col(dbc.Input(id="rs-z-score-min", type="number", placeholder="Min")),
                dbc.Col(dbc.Input(id="rs-z-score-max", type="number", placeholder="Max"))
            ])
        ]),
        dbc.Col([
            html.Label("Volume Z-Score Min"),
            dbc.Input(id="volume-std-dev-min", type="number", placeholder="Min std deviations above avg")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Avg 30-Day Volume Min"),
            dbc.Input(id="avg-volume-min", type="number", placeholder="Min shares")
        ]),
        dbc.Col([
            html.Label("Weinstein Stage"),
            dbc.Select(
                id="weinstein-stage",
                options=[
                    {"label": "All Stages", "value": None},
                    {"label": "Stage 1: Basing", "value": "Stage 1"},
                    {"label": "Stage 2: Advancing", "value": "Stage 2"},
                    {"label": "Stage 3: Topping", "value": "Stage 3"},
                    {"label": "Stage 4: Declining", "value": "Stage 4"}
                ],
                value=None
            )
        ])
    ], className="mt-3"),
    dbc.Row([
        dbc.Col(
            dbc.Button("Search", id="search-button", color="primary", className="mt-3")
        )
    ])
])

# ETF Relative Strength Tab Components
import dash_table

def get_etf_grid_data():
    df = fetch_latest_etf_metrics()
    if df.empty:
        return [], []
    columns = [
        {"name": "ETF Ticker", "id": "etf_ticker", "presentation": "markdown"},
        {"name": "ETF Name", "id": "etf_name"},
        {"name": "RS Rank Index", "id": "rs_rank_index"},
        {"name": "RS Z-Score", "id": "rs_z_score"}
    ]
    data = df.to_dict('records')
    return columns, data

etf_grid_columns, etf_grid_data = get_etf_grid_data()

etf_grid = dash_table.DataTable(
    id='etf-grid',
    columns=etf_grid_columns,
    data=etf_grid_data,
    sort_action='native',
    row_selectable='single',
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left'},
    style_header={'fontWeight': 'bold'},
    page_size=20
)

etf_price_chart = dcc.Graph(id='etf-price-chart')
etf_metrics_chart = dcc.Graph(id='etf-metrics-chart')

# Add date range selector for ETF tab
etf_date_range = dcc.DatePickerRange(
    id='etf-date-range',
    start_date=(datetime.now() - timedelta(days=365)).date(),
    end_date=datetime.now().date(),
    display_format='YYYY-MM-DD',
    className='mb-3'
)

# Add separate graphs for each metric
etf_rs_chart = dcc.Graph(id='etf-rs-chart')
etf_rs_rank_chart = dcc.Graph(id='etf-rs-rank-chart')
etf_zscore_chart = dcc.Graph(id='etf-zscore-chart')

# Option Data Tab Components
option_symbol_dropdown = dcc.Dropdown(
    id='option-symbol-dropdown',
    options=[{'label': s, 'value': s} for s in get_all_option_symbols()],
    placeholder="Select option symbol...",
    style={"width": "300px"}
)

option_data_table = dash_table.DataTable(
    id='option-data-table',
    columns=[],  # Will be set dynamically
    data=[],
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left', 'minWidth': '100px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
    style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230,230,230)'},
    page_size=30,
    active_cell=None
)

option_metric_graph = dcc.Graph(id='option-metric-graph')

# Option Data Screener Tab Components
option_metric_dropdown = dcc.Dropdown(
    id='option-metric-dropdown',
    options=[{'label': col, 'value': col} for col in get_option_numerical_columns()],
    placeholder="Select metric to screen...",
    style={"width": "300px"}
)

option_min_multiple_input = dbc.Input(
    id='option-min-multiple',
    type='number',
    placeholder="Minimum multiple (e.g., 5.0)",
    min=0,
    step=0.1,
    style={"width": "200px"}
)

option_screener_button = dbc.Button(
    "Search",
    id="option-screener-button",
    color="primary",
    className="ml-2"
)

option_screener_table = dash_table.DataTable(
    id='option-screener-table',
    columns=[
        {"name": "Symbol", "id": "symbol"},
        {"name": "Current Value", "id": "current_value"},
        {"name": "30-Day Avg", "id": "avg_value"},
        {"name": "Multiple", "id": "multiple"}
    ],
    data=[],
    sort_action='native',
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'left', 'minWidth': '100px', 'maxWidth': '200px'},
    style_header={'fontWeight': 'bold', 'backgroundColor': 'rgb(230,230,230)'},
    page_size=50
)

# Main layout
app.layout = dbc.Container([
    html.H1("Stock Analysis Dashboard", className="my-4"),
    
    dbc.Tabs([
        dbc.Tab(label="Single Stock Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Stock"),
                    ticker_dropdown
                ], width=3),
                dbc.Col([
                    html.Label("ETF"),
                    etf_dropdown
                ], width=3),
                dbc.Col([
                    html.Label("Date Range"),
                    date_range
                ], width=3),
                dbc.Col([
                    html.Label("Interval"),
                    interval_dropdown
                ], width=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='price-chart')
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='metrics-chart')
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='rs-ma-chart')
                ], width=12)
            ])
        ]),
        
        dbc.Tab(label="Stock Screener", children=[
            html.Div([
                html.H3("Filter Stocks", className="my-4"),
                html.P("Volume Z-Score: Shows stocks where current volume is X standard deviations above the 1-year average volume.", 
                       className="text-muted mb-3"),
                metrics_filters,
                html.Div(id="search-results", className="mt-4")
            ])
        ]),
        dbc.Tab(label="ETF Relative Strength", children=[
            html.H3("ETF Relative Strength Metrics", className="my-4"),
            etf_date_range,
            etf_grid,
            html.Div(id='etf-selected-label', className='mt-4'),
            etf_rs_chart,
            etf_rs_rank_chart,
            etf_zscore_chart
        ]),
        dbc.Tab(label="Option Data", children=[
            html.H3("Option Data: Underlying Metrics", className="my-4"),
            html.Div([
                html.Label("Select Symbol:"),
                option_symbol_dropdown
            ], className="mb-3"),
            html.Div([
                option_data_table
            ], className="mb-4"),
            html.Div([
                option_metric_graph
            ])
        ]),
        dbc.Tab(label="Option Data Screener", children=[
            html.H3("Option Data Screener", className="my-4"),
            html.P("Find symbols where current values are significantly above their 30-day averages.", 
                   className="text-muted mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Metric:"),
                    option_metric_dropdown
                ], width=4),
                dbc.Col([
                    html.Label("Minimum Multiple:"),
                    option_min_multiple_input
                ], width=4),
                dbc.Col([
                    html.Label("&nbsp;"),  # Spacer
                    option_screener_button
                ], width=4, className="d-flex align-items-end")
            ], className="mb-4"),
            html.Div([
                option_screener_table
            ])
        ])
    ])
], fluid=True)

def resample_ohlcv(df, interval):
    if interval == 'D':
        return df
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df.resample(interval, label='right', closed='right').agg(ohlc_dict)
    df_resampled = df_resampled.dropna(subset=['open', 'high', 'low', 'close'])
    return df_resampled

@app.callback(
    Output('etf-select', 'options'),
    Output('etf-select', 'value'),
    Input('ticker-select', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_etf_dropdown(ticker, start_date, end_date):
    if not ticker:
        return [], None
    metrics_data = fetch_stock_metrics(ticker, start_date, end_date)
    etfs = metrics_data['etf_ticker'].dropna().unique()
    options = [{'label': etf, 'value': etf} for etf in etfs]
    value = etfs[0] if len(etfs) == 1 else None
    return options, value

@app.callback(
    [Output('price-chart', 'figure'),
     Output('metrics-chart', 'figure')],
    [Input('ticker-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('interval-select', 'value'),
     Input('etf-select', 'value')]
)
def update_charts(ticker, start_date, end_date, interval, etf_ticker):
    if not ticker:
        return {}, {}
    
    # Fetch data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    metrics_data = fetch_stock_metrics(ticker, start_date, end_date)
    
    if etf_ticker:
        metrics_data = metrics_data[metrics_data['etf_ticker'] == etf_ticker]
    
    if stock_data.empty:
        return {}, {}
    
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data.set_index('date', inplace=True)
    stock_data = resample_ohlcv(stock_data, interval)
    stock_data.reset_index(inplace=True)
    
    # Color volume bars
    volume_colors = [
        'green' if c >= o else 'red'
        for o, c in zip(stock_data['open'], stock_data['close'])
    ]
    
    # Create candlestick chart
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig1.add_trace(
        go.Candlestick(
            x=stock_data['date'],
            open=stock_data['open'],
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Bar(
            x=stock_data['date'],
            y=stock_data['volume'],
            marker_color=volume_colors,
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Remove all non-trading days (weekends + holidays) from x-axis
    all_dates = pd.date_range(stock_data['date'].min(), stock_data['date'].max())
    weekends = all_dates[all_dates.weekday >= 5]  # 5=Saturday, 6=Sunday
    all_bdays = pd.bdate_range(stock_data['date'].min(), stock_data['date'].max())
    holidays = all_bdays.difference(stock_data['date'])
    all_non_trading_days = weekends.union(holidays)

    fig1.update_layout(
        title=f"{ticker} Price and Volume",
        xaxis_rangeslider_visible=False,
        height=600,
        xaxis=dict(
            rangebreaks=[
                dict(values=all_non_trading_days)
            ]
        )
    )
    
    # Calculate moving averages for relative strength
    metrics_data['MA252'] = metrics_data['relative_strength'].rolling(window=252, min_periods=1).mean()
    metrics_data['MA150'] = metrics_data['relative_strength'].rolling(window=150, min_periods=1).mean()
    
    # Create metrics chart
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Relative Strength', 'RS Ranks', 'RS Z-Score'))
    
    # Add relative strength and its moving averages
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['relative_strength'],
            name='Relative Strength',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['MA252'],
            name='252-day MA',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['MA150'],
            name='150-day MA',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['rs_rank_index'],
            name='RS Rank (Index)',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['rs_rank_overall'],
            name='RS Rank (Overall)',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=metrics_data['date'],
            y=metrics_data['rs_z_score'],
            name='RS Z-Score',
            line=dict(color='brown')
        ),
        row=3, col=1
    )
    
    # Remove all non-trading days (weekends + holidays) from x-axis for metrics chart
    all_dates_metrics = pd.date_range(metrics_data['date'].min(), metrics_data['date'].max())
    weekends_metrics = all_dates_metrics[all_dates_metrics.weekday >= 5]
    all_bdays_metrics = pd.bdate_range(metrics_data['date'].min(), metrics_data['date'].max())
    holidays_metrics = all_bdays_metrics.difference(metrics_data['date'])
    all_non_trading_days_metrics = weekends_metrics.union(holidays_metrics)

    fig2.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangebreaks=[
                dict(values=all_non_trading_days_metrics)
            ]
        )
    )
    
    fig2.update_yaxes(title_text="Relative Strength", row=1, col=1)
    fig2.update_yaxes(title_text="Rank %", row=2, col=1)
    fig2.update_yaxes(title_text="Z-Score", row=3, col=1)
    
    return fig1, fig2

@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('etf-filter', 'value'),
     State('rs-rank-min', 'value'),
     State('rs-rank-max', 'value'),
     State('rs-rank-overall-min', 'value'),
     State('rs-rank-overall-max', 'value'),
     State('rs-z-score-min', 'value'),
     State('rs-z-score-max', 'value'),
     State('volume-std-dev-min', 'value'),
     State('avg-volume-min', 'value'),
     State('weinstein-stage', 'value')]
)
def update_search_results(n_clicks, etf_ticker, rs_rank_min, rs_rank_max, 
                         rs_rank_overall_min, rs_rank_overall_max,
                         rs_z_score_min, rs_z_score_max,
                         volume_std_dev_min, avg_volume_min,
                         weinstein_stage):
    if n_clicks is None:
        return []
        
    def safe_float(value):
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
            
    def percentage_to_decimal(value):
        if value is not None:
            return value / 100
        return None
        
    # Convert inputs to appropriate types
    rs_rank_min = safe_float(rs_rank_min)
    rs_rank_max = safe_float(rs_rank_max)
    rs_rank_overall_min = safe_float(rs_rank_overall_min)
    rs_rank_overall_max = safe_float(rs_rank_overall_max)
    rs_z_score_min = safe_float(rs_z_score_min)
    rs_z_score_max = safe_float(rs_z_score_max)
    volume_std_dev_min = safe_float(volume_std_dev_min)
    avg_volume_min = safe_float(avg_volume_min)
    
    # Search for stocks
    results = search_stocks_by_metrics(
        rs_rank_min=rs_rank_min,
        rs_rank_max=rs_rank_max,
        rs_rank_overall_min=rs_rank_overall_min,
        rs_rank_overall_max=rs_rank_overall_max,
        rs_z_score_min=rs_z_score_min,
        rs_z_score_max=rs_z_score_max,
        volume_std_dev_min=volume_std_dev_min,
        avg_volume_min=avg_volume_min,
        weinstein_stage=weinstein_stage,
        etf_ticker=etf_ticker
    )
    
    if results.empty:
        return html.Div("No stocks found matching the criteria.")
        
    def format_percentage(x):
        return f"{x * 100:.1f}%" if pd.notnull(x) else "N/A"
        
    def format_float(x):
        return f"{x:.2f}" if pd.notnull(x) else "N/A"
        
    def format_volume(x):
        return f"{x:,.0f}" if pd.notnull(x) else "N/A"
        
    # Format the results
    results['RS Rank'] = results['rs_rank_index'].apply(format_percentage)
    results['RS Rank Overall'] = results['rs_rank_overall'].apply(format_percentage)
    results['RS Z-Score'] = results['rs_z_score'].apply(format_float)
    results['Volume Z-Score'] = results['volume_z_score'].apply(format_float)
    results['Avg 30-Day Volume'] = results['avg_30d_volume'].apply(format_volume)
    
    # Create the table
    table = dash_table.DataTable(
        data=results.to_dict('records'),
        columns=[
            {'name': 'Ticker', 'id': 'ticker'},
            {'name': 'ETF', 'id': 'etf_ticker'},
            {'name': 'RS Rank', 'id': 'RS Rank'},
            {'name': 'RS Rank Overall', 'id': 'RS Rank Overall'},
            {'name': 'RS Z-Score', 'id': 'RS Z-Score'},
            {'name': 'Volume Z-Score', 'id': 'Volume Z-Score'},
            {'name': 'Avg 30-Day Volume', 'id': 'Avg 30-Day Volume'}
        ],
        sort_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return table

@app.callback(
    [Output('etf-selected-label', 'children'),
     Output('etf-rs-chart', 'figure'),
     Output('etf-rs-rank-chart', 'figure'),
     Output('etf-zscore-chart', 'figure')],
    [Input('etf-grid', 'selected_rows'),
     Input('etf-date-range', 'start_date'),
     Input('etf-date-range', 'end_date')]
)
def update_etf_charts(selected_rows, start_date, end_date):
    columns, data = get_etf_grid_data()
    if not selected_rows or not data:
        return "Select an ETF to view details.", go.Figure(), go.Figure(), go.Figure()
    row = data[selected_rows[0]]
    etf_ticker = row['etf_ticker']
    etf_name = row.get('etf_name', '')
    # Fetch metrics for selected date range
    metrics = fetch_etf_metrics(etf_ticker, start_date, end_date)
    if metrics.empty:
        return f"No data for {etf_ticker}.", go.Figure(), go.Figure(), go.Figure()
    # Relative Strength with 30-week MA
    rs_fig = go.Figure()
    rs_fig.add_trace(go.Scatter(
        x=metrics['date'],
        y=metrics['relative_strength'],
        mode='lines',
        name='Relative Strength'))
    # 30-week MA (150 trading days)
    if len(metrics) >= 150:
        rs_ma = metrics['relative_strength'].rolling(window=150, min_periods=1).mean()
        rs_fig.add_trace(go.Scatter(
            x=metrics['date'],
            y=rs_ma,
            mode='lines',
            name='30-Week MA',
            line=dict(dash='dash', color='orange')
        ))
    rs_fig.update_layout(
        title=f"{etf_ticker} ({etf_name}) Relative Strength",
        yaxis_title='Relative Strength',
        xaxis_title='Date',
        legend=dict(orientation='h'),
        height=350
    )
    # RS Rank Index
    rs_rank_fig = go.Figure()
    rs_rank_fig.add_trace(go.Scatter(
        x=metrics['date'],
        y=metrics['rs_rank_index'],
        mode='lines',
        name='RS Rank Index'))
    rs_rank_fig.update_layout(
        title=f"{etf_ticker} ({etf_name}) RS Rank Index",
        yaxis_title='RS Rank Index',
        xaxis_title='Date',
        legend=dict(orientation='h'),
        height=350
    )
    # RS Z-Score
    zscore_fig = go.Figure()
    zscore_fig.add_trace(go.Scatter(
        x=metrics['date'],
        y=metrics['rs_z_score'],
        mode='lines',
        name='RS Z-Score'))
    zscore_fig.update_layout(
        title=f"{etf_ticker} ({etf_name}) RS Z-Score",
        yaxis_title='RS Z-Score',
        xaxis_title='Date',
        legend=dict(orientation='h'),
        height=350
    )
    label = f"Selected ETF: {etf_ticker} ({etf_name})"
    return label, rs_fig, rs_rank_fig, zscore_fig

# Add 30-week MA to Single Stock Analysis tab's metrics chart
@app.callback(
    Output('rs-ma-chart', 'figure'),
    [Input('ticker-select', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('etf-select', 'value')]
)
def update_stock_rs_ma_chart(ticker, start_date, end_date, etf_ticker):
    if not ticker:
        return go.Figure()
    metrics_data = fetch_stock_metrics(ticker, start_date, end_date)
    if etf_ticker:
        metrics_data = metrics_data[metrics_data['etf_ticker'] == etf_ticker]
    if metrics_data.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_data['date'],
        y=metrics_data['relative_strength'],
        mode='lines',
        name='Relative Strength'))
    # 30-week MA (150 trading days)
    if len(metrics_data) >= 150:
        rs_ma = metrics_data['relative_strength'].rolling(window=150, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=metrics_data['date'],
            y=rs_ma,
            mode='lines',
            name='30-Week MA',
            line=dict(dash='dash', color='orange')
        ))
    fig.update_layout(
        title="Relative Strength (with 30-Week MA)",
        yaxis_title='Relative Strength',
        xaxis_title='Date',
        legend=dict(orientation='h'),
        height=350
    )
    return fig

# Option Data Tab Callbacks
@app.callback(
    [Output('option-data-table', 'columns'),
     Output('option-data-table', 'data'),
     Output('option-data-table', 'active_cell')],
    [Input('option-symbol-dropdown', 'value')]
)
def update_option_data_table(symbol):
    if not symbol:
        return [], [], None
    # Get last 30 days of data
    df = fetch_option_last_30_days(symbol)
    if df.empty:
        return [], [], None
    # Get ratio row
    ratio_dict = fetch_option_latest_comparison(symbol)
    # Prepare columns
    columns = [
        {"name": col, "id": col} for col in df.columns if col not in ['created_at']
    ]
    # Prepare data: first row is ratio, then 30 days of data
    ratio_row = {col: (f"{ratio_dict.get(col):.2f}" if col in ratio_dict and pd.notnull(ratio_dict[col]) else "") for col in df.columns if col not in ['created_at']}
    ratio_row['collection_date'] = 'Latest/Avg Ratio'
    ratio_row['symbol'] = symbol
    data = [ratio_row] + df.to_dict('records')
    return columns, data, None

@app.callback(
    Output('option-metric-graph', 'figure'),
    [Input('option-data-table', 'active_cell'),
     Input('option-symbol-dropdown', 'value'),
     State('option-data-table', 'columns')]
)
def update_option_metric_graph(active_cell, symbol, columns):
    import plotly.graph_objs as go
    if not symbol or not columns:
        return go.Figure()
    # Only update if header cell is clicked
    if not active_cell or active_cell.get('row') != 0:
        return go.Figure()
    col_id = active_cell.get('column_id')
    if not col_id or col_id in ['collection_date', 'symbol']:
        return go.Figure()
    # Fetch full history for this column
    df_data, df_avg = fetch_option_column_history(symbol, col_id)
    if df_data.empty or df_avg.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_data['collection_date'],
        y=df_data[col_id],
        mode='lines',
        name=f'{col_id} (actual)'
    ))
    fig.add_trace(go.Scatter(
        x=df_avg['date'],
        y=df_avg[col_id + '_avg'] if (col_id + '_avg') in df_avg.columns else df_avg.iloc[:,1],
        mode='lines',
        name=f'{col_id} (30d avg)',
        line=dict(dash='dash', color='orange')
    ))
    fig.update_layout(
        title=f"{symbol}: {col_id} vs 30d Avg",
        yaxis_title=col_id,
        xaxis_title='Date',
        legend=dict(orientation='h'),
        height=400
    )
    return fig

# Option Data Screener Callback
@app.callback(
    Output('option-screener-table', 'data'),
    [Input('option-screener-button', 'n_clicks')],
    [State('option-metric-dropdown', 'value'),
     State('option-min-multiple', 'value')]
)
def update_option_screener_results(n_clicks, metric, min_multiple):
    if n_clicks is None or not metric or min_multiple is None:
        return []
    
    try:
        min_multiple = float(min_multiple)
    except (ValueError, TypeError):
        return []
    
    results = fetch_option_screener_results(metric, min_multiple)
    if results.empty:
        return []
    
    # Format the data for display
    formatted_data = []
    for _, row in results.iterrows():
        formatted_data.append({
            'symbol': row['symbol'],
            'current_value': f"{row['current_value']:,.2f}" if pd.notnull(row['current_value']) else "N/A",
            'avg_value': f"{row['avg_value']:,.2f}" if pd.notnull(row['avg_value']) else "N/A",
            'multiple': f"{row['multiple']:.2f}" if pd.notnull(row['multiple']) else "N/A"
        })
    
    return formatted_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
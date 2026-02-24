"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       PORTFOLIO ALPHA AND BETA PROJECT - NVDA vs SPY Analysis                ║
║       Using Interactive Brokers API (IBAPI)                                  ║
║       CAPM Regression & Risk Decomposition                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

This project decomposes NVDA returns into:
  • Beta (β): Systematic Risk / Market Exposure
  • Alpha (α): Idiosyncratic Return / True Performance ("Edge")

Using the CAPM framework:
  R_NVDA - R_f = α + β(R_SPY - R_f) + ε

"""

import threading
import time
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for elegant plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['font.family'] = 'DejaVu Sans'

#=================================================================
# IBAPI CONNECTION AND DATA FETCHING
#=================================================================

class IBDataFetcher(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = defaultdict(list)
        self.data_received = threading.Event()
        self.connection_established = threading.Event()
        self.next_order_id = None
        self.current_req_id = None
    
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158, 2119]:
            return
        if errorCode == 200:
            print(f" ⚠️ Contract not found or no data available")
        elif self.error not in [2104, 2106, 2158]:
            print(f" ⚠️ Error {errorCode}: {errorString}")
    
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        self.connection_established.set()

    def historicalData(self, reqId, bar):
        self.data[reqId].append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })
    
    def historicalDataEnd(self, reqId, start, end):
        self.data_received.set()
    
def create_contract(symbol, exchange="SMART", primary_exchange="NASDAQ"):
    """Create stock contract specification."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = exchange
    contract.currency = "USD"
    contract.primaryExchange = primary_exchange
    return contract

def fetch_historical_data(app, contract, duration="5 Y", bar_size="1 day"):
    """Fetch historical data from IB"""
    req_id = app.next_order_id if app.next_order_id else 1
    app.next_order_id = req_id + 1
    app.data[req_id] = []
    app.data_received.clear()

    end_datetime = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_datetime,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    if not app.data_received.wait(timeout=60):
        print(" ⚠️ timeout waiting for historical data")
        return pd.DataFrame()
    
    if not app.data[req_id]:
        return pd.DataFrame()
    
    df = pd.DataFrame(app.data[req_id])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    return df


#======================================================================
# DATA PROCESSING AND RETURNS CALCULATION
#======================================================================
def calculate_returns(df, column='Close'):
    """
    Calculate daily simple returns.

    R_t = (P_t - P_{t-1} / P_{t-1})
    """
    df.copy()
    df['Return'] = df[column].pct_change()
    df = df.dropna(subset=['Return'])
    return df

def merge_datasets(nvda_df, spy_df):
    """
    Merge NVDA and SPY datasets on Date.
    Ensures we only use dates where both assets are traded.
    """
    nvda = nvda_df[['Date', 'Close', 'Return']].copy()
    nvda.columns = ['Date', 'NVDA_Close', 'NVDA_Return']

    spy = spy_df[['Date', 'Close', 'Return']].copy()
    spy.columns = ['Date', 'SPY_Close', 'SPY_Return']

    merged = pd.merge(nvda, spy, on='Date', how='inner')
    return merged


#==================================================================
# REGRESSION MODEL (OLS Implementation)
#==================================================================

def calculate_beta_alpha(y, x):
    """
    Calculate Beta and Alpha using OLS formulas.

    CAPM Model: R_NVDA = α + β x R_SPY + ε

    β̂ = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
    α̂ = ȳ - β̂ x x̄
    
    Where:
    - y = NVDA returns (dependent variable)
    - x = SPY returns (independent variable / market factor)
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Covariance(x,y) / variance(x)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    beta = numerator / denominator
    alpha = y_mean - (beta * x_mean)

    return beta, alpha

def calculate_regression_metrics(y_true, y_pred, n_params=2):
    """
    Calculate regression quality metrics
    """
    residuals = y_true - y_pred

    # Mean Squared Error
    mse = np.mean(residuals ** 2)

    # R-squared (coefficient of determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Adjusted R-squared
    n = len(y_true)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params)

    # Standard error of beta
    sigma_squared = ss_res / (n - n_params)
    x_mean = np.mean(y_pred)    #This is a simplification

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'residual_std': np.std(residuals),
        'n_observations': n
    }

def calculate_rolling_beta(df, window_days=126):
    """
    Calculate rolling beta using a sliding window.

    Window: 126 Trading days = 6 Months

    This reveals non-stationarity in the beta parameter.
    """

    rolling_dates = []
    rolling_betas = []
    rolling_alphas = []

    for i in range(window_days, len(df)):
        window = df.iloc[i-window_days:i]

        y = window['NVDA_Return'].values
        x = window['SPY_Return'].values

        beta, alpha = calculate_beta_alpha(y, x)

        rolling_betas.append(beta)
        rolling_alphas.append(alpha)
        rolling_dates.append(df.iloc[i]['Date'])

    return pd.DataFrame({
        'Date': rolling_dates,
        'Rolling_Beta': rolling_betas,
        'Rolling_Alpha': rolling_alphas
    })


#=================================================================
# MATPLOTLIB VISUALISATIONS
#=================================================================

def create_regression_plot(df, beta, alpha, metrics):
    """
    Create a scatter plot of NVDA returns vs SPY returns with regression line.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df['SPY_Return'].values * 100 # Convert to percentage
    y = df['NVDA_Return'].values * 100

    # Scatter plot with colour based density
    scatter = ax.scatter(x, y, alpha=0.5, c='#00d9ff', s=20, edgecolors='none')

    # Regression Line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = alpha * 100 + beta * x_line
    ax.plot(x_line, y_line, color='#e94560', linewidth=3, label=f'Regression: NVDA = {alpha*100:.4f} + {beta:.2f} x SPY')

    # Reference Line (beta = 1)
    ax.plot(x_line, x_line, color='#ffd700', linewidth=1.5, linestyle='--', alpha=0.7, label='β = 1 Reference')

    # Zero Lines
    ax.axhline(y=0, color='#eaeaea', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='#eaeaea', linewidth=0.5, alpha=0.5)

    # Labels and title
    ax.set_xlabel('SPY Daily Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NVDA Daily Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('CAPM Regression: NVDA Reutrns vs SPY Returns\n' + f'β = {beta:.3f} | α (daily) = {alpha*100:.4f}% | R² = {metrics["r_squared"]:.3f}',
                 fontsize=14, fontweight='bold', pad=20) 
    
    # Add annotation box with key stats:
    stats_text = (f"Key Statistics:\n"
                  f"-----------------\n"
                  f"Beta (β):     {beta:.3f}\n"
                  f"Alpha (α):    {alpha*252*100:.2f}% ann.\n"
                  f"R-squared:    {metrics['r_squared']:.3f}\n"
                  f"Observations: {metrics['n_observations']:,}")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='#0f3460', edgecolor='#e94560', alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)
    
    ax.legend(loc='lower right', fontsize=10)

    # Grid Styling
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_rolling_beta_plot(rolling_df):
    """
    Create a plot showing rolling beta over time.
    Reveals non-stationarity in the NVDA-SPY Relationship.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    dates = rolling_df['Date']
    betas = rolling_df['Rolling_Beta']
    alphas = rolling_df['Rolling_Alpha'] * 252 * 100  # Annualized %
    
    # Plot 1: Rolling Beta
    ax1.fill_between(dates, betas, 1, where=(betas > 1), alpha=0.3, color='#e94560', label='β > 1 (More volatile)')
    ax1.fill_between(dates, betas, 1, where=(betas <= 1), alpha=0.3, color='#00d9ff', label='β ≤ 1 (Less volatile)')
    ax1.plot(dates, betas, color='#ffffff', linewidth=2)
    
    # Reference lines
    ax1.axhline(y=1, color='#ffd700', linewidth=2, linestyle='--', label='β = 1 (Market)')
    ax1.axhline(y=betas.mean(), color='#00ff88', linewidth=1.5, linestyle=':', label=f'Mean β = {betas.mean():.2f}')
    
    ax1.set_ylabel('Rolling Beta (6-month)', fontsize=12, fontweight='bold')
    ax1.set_title('NVDA Rolling Beta Over Time\nRevealing Non-Stationarity in Market Exposure',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add key event annotations
    ax1.annotate('Higher β = More\nmarket exposure', xy=(dates.iloc[len(dates)//4], betas.max()),
                fontsize=9, color='#e94560', ha='center')
    
    # Plot 2: Rolling Alpha (Annualised)
    colours = ['#00ff88' if a > 0 else '#e94560' for a in alphas]
    ax2.bar(dates, alphas, color=colours, alpha=0.7, width=2)
    ax2.axhline(y=0, color='#eaeaea', linewidth=1)
    ax2.axhline(y=alphas.mean(), color='#ffd700', linewidth=1.5, linestyle='--', 
                label=f'Mean α = {alphas.mean():.1f}% ann.')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rolling Alpha (% ann.)', fontsize=12, fontweight='bold')
    ax2.set_title('Rolling Alpha: NVDA\'s Market-Independent Performance', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig

def create_cumulative_returns_plot(df):
    """
    Create cumulative returns comparison plot.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate cumulative returns
    nvda_cum = (1 + df['NVDA_Return']).cumprod()
    spy_cum = (1 + df['SPY_Return']).cumprod()
    
    ax.plot(df['Date'], nvda_cum, color='#00ff88', linewidth=2, label=f'NVDA: {(nvda_cum.iloc[-1]-1)*100:.1f}%')
    ax.plot(df['Date'], spy_cum, color='#e94560', linewidth=2, label=f'SPY: {(spy_cum.iloc[-1]-1)*100:.1f}%')
    
    ax.fill_between(df['Date'], nvda_cum, spy_cum, where=(nvda_cum > spy_cum), 
                    alpha=0.3, color='#00ff88', label='NVDA Outperformance')
    ax.fill_between(df['Date'], nvda_cum, spy_cum, where=(nvda_cum <= spy_cum), 
                    alpha=0.3, color='#e94560', label='SPY Outperformance')
    
    ax.axhline(y=1, color='#eaeaea', linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (1 = Starting Value)', fontsize=12, fontweight='bold')
    ax.set_title('NVDA vs SPY: Cumulative Returns Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    return fig

#=======================================================================
# TERMINAL DISPLAY
#=======================================================================
def print_header():
    """Print the project header."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  📈 PORTFOLIO ALPHA AND BETA PROJECT - NVDA vs SPY".center(78) + "║")
    print("║" + "  CAPM Regression & Risk Decomposition Analysis".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_section_header(title, emoji=""):
    """Print a formatted section header."""
    print("\n" + "─" * 80)
    print(f"  {emoji} {title}")
    print("─" * 80)


def print_data_summary(nvda_df, spy_df, merged_df):
    """Display summary statistics about the data."""
    print_section_header("DATA SUMMARY", "📊")
    
    # Box width
    w = 74
    
    # NVDA stats
    nvda_ret = merged_df['NVDA_Return']
    nvda_total = (1 + nvda_ret).prod() - 1
    nvda_annual = ((1 + nvda_total) ** (252 / len(nvda_ret))) - 1
    
    # SPY stats  
    spy_ret = merged_df['SPY_Return']
    spy_total = (1 + spy_ret).prod() - 1
    spy_annual = ((1 + spy_total) ** (252 / len(spy_ret))) - 1
    
    print()
    print("  ┌" + "─" * w + "┐")
    print(f"  │{'NVDA (The Asset)':^{w}}│")
    print("  ├" + "─" * w + "┤")
    print(f"  │  • Date Range:       {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d'):<{w-47}}│")
    print(f"  │  • Trading Days:     {len(merged_df):,} days{'':<{w-30}}│")
    print(f"  │  • Total Return:     {nvda_total*100:+.2f}%{'':<{w-29}}│")
    print(f"  │  • Annualised Return:{nvda_annual*100:+.2f}%{'':<{w-29}}│")
    print(f"  │  • Daily Volatility: {nvda_ret.std()*100:.2f}%{'':<{w-29}}│")
    print(f"  │  • Ann. Volatility:  {nvda_ret.std()*np.sqrt(252)*100:.2f}%{'':<{w-29}}│")
    print("  └" + "─" * w + "┘")
    
    print()
    print("  ┌" + "─" * w + "┐")
    print(f"  │{'SPY (The Market Proxy)':^{w}}│")
    print("  ├" + "─" * w + "┤")
    print(f"  │  • Total Return:     {spy_total*100:+.2f}%{'':<{w-29}}│")
    print(f"  │  • Annualised Return:{spy_annual*100:+.2f}%{'':<{w-29}}│")
    print(f"  │  • Daily Volatility: {spy_ret.std()*100:.2f}%{'':<{w-29}}│")
    print(f"  │  • Ann. Volatility:  {spy_ret.std()*np.sqrt(252)*100:.2f}%{'':<{w-29}}│")
    print("  └" + "─" * w + "┘")
    
    # Outperformance
    outperf = nvda_total - spy_total
    print()
    print(f"  📊 NVDA vs SPY Outperformance: {outperf*100:+.2f}% (total period)")
    print()


def print_regression_results(beta, alpha, metrics):
    """Display regression results with interpretation."""
    print_section_header("REGRESSION RESULTS (CAPM)", "📐")
    
    alpha_ann = alpha * 252 * 100  # Annualised alpha in %
    
    print(f"""
  ╭────────────────────────────────────────────────────────────────────────────╮
  │  THE CAPM MODEL                                                            │
  │                                                                            │
  │  R_NVDA = α + β × R_SPY + ε                                                │
  │                                                                            │
  │  Where:                                                                    │
  │    • R_NVDA = NVDA daily return (dependent variable)                       │
  │    • R_SPY  = SPY daily return (market factor)                             │
  │    • α (Alpha) = Market-independent return ("Edge")                        │
  │    • β (Beta)  = Systematic risk / market exposure                         │
  │    • ε = Residual / unexplained variance                                   │
  ╰────────────────────────────────────────────────────────────────────────────╯
""")
        
    w = 74
    print("  ┌" + "─" * w + "┐")
    print(f"  │{'ESTIMATED PARAMETERS':^{w}}│")
    print("  ├" + "─" * w + "┤")
    print(f"  │  β (Beta):          {beta:>10.4f}  {'━━━ Systematic Risk Coefficient':<{w-34}}│")
    print(f"  │  α (Alpha daily):   {alpha*100:>+10.6f}%{'━━━ Daily excess return':<{w-34}}│")
    print(f"  │  α (Alpha annual):  {alpha_ann:>+10.2f}% {'━━━ Annualised: α x 252':<{w-34}}│")
    print("  ├" + "─" * w + "┤")
    print(f"  │{'MODEL FIT':^{w}}│")
    print("  ├" + "─" * w + "┤")
    print(f"  │  R² (R-squared):    {metrics['r_squared']:>10.4f}  {'━━━ Variance explained by market':<{w-34}}│")
    print(f"  │  Adj. R²:           {metrics['adj_r_squared']:>10.4f}  {'━━━ Adjusted for parameters':<{w-34}}│")
    print(f"  │  MSE:               {metrics['mse']:>10.6f}  {'━━━ Mean Squared Error':<{w-34}}│")
    print(f"  │  RMSE:              {metrics['rmse']*100:>10.4f}% {'━━━ Root MSE (in return units)':<{w-34}}│")
    print("  └" + "─" * w + "┘")
    print()

def print_beta_interpretation(beta, metrics):
    """Interpret the beta value."""
    print_section_header("INTERPRETING BETA (β)", "🎯")
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  YOUR RESULT: β = {beta:.3f}                                                    │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │""")
    
    if beta > 1:
        leverage_equiv = beta
        print(f"""  │  📈 β > 1: NVDA is MORE volatile than the market                           │
  │                                                                            │
  │  Interpretation:                                                           │
  │    • For every 1% move in SPY, NVDA tends to move {beta:.2f}%                    │
  │    • NVDA amplifies market movements by {(beta-1)*100:.0f}%                             │
  │    • Holding NVDA ≈ Holding a {leverage_equiv:.1f}x LEVERAGED S&P 500 position            │
  │                                                                            │
  │  Risk Implication:                                                         │
  │    • Higher potential returns in bull markets                              │
  │    • Higher potential losses in bear markets                               │
  │    • More volatile equity curve                                            │""")
    elif beta < 1:
        print(f"""  │  📉 β < 1: NVDA is LESS volatile than the market                           │
  │                                                                            │
  │  Interpretation:                                                           │
  │    • For every 1% move in SPY, NVDA tends to move {beta:.2f}%                    │
  │    • NVDA dampens market movements by {(1-beta)*100:.0f}%                               │
  │    • NVDA acts as a defensive position relative to market                  │
  │                                                                            │
  │  Risk Implication:                                                         │
  │    • Lower potential returns in bull markets                               │
  │    • Lower potential losses in bear markets                                │
  │    • More stable equity curve                                              │""")
    else:
        print(f"""  │  ⚖️  β ≈ 1: NVDA moves in line with the market                              │
  │                                                                            │
  │  Interpretation:                                                           │
  │    • NVDA and SPY move roughly 1:1                                         │
  │    • No leverage effect from beta exposure                                 │""")
    
    print(f"""  │                                                                            │
  │  R² = {metrics['r_squared']:.1%} means the market explains {metrics['r_squared']*100:.1f}% of NVDA's variance          │
  │  The remaining {(1-metrics['r_squared'])*100:.1f}% is due to NVDA-specific factors                    │
  └────────────────────────────────────────────────────────────────────────────┘
""")

def print_alpha_interpretation(alpha, beta):
    """Interpret the alpha value."""
    print_section_header("THE SEARCH FOR ALPHA (α)", "💰")
    
    alpha_daily = alpha * 100
    alpha_annual = alpha * 252 * 100
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  YOUR RESULT: α = {alpha_daily:+.6f}% daily ({alpha_annual:+.2f}% annualised)                  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │                                                                            │""")
    
    if alpha > 0:
        print(f"""  │  ✅ α > 0: NVDA generated POSITIVE alpha                                   │
  │                                                                            │
  │  Interpretation:                                                           │
  │    • NVDA outperformed what its market exposure would predict              │
  │    • {alpha_annual:.2f}% annual return came from company-specific excellence          │
  │    • This is "true" outperformance, not just beta exposure                 │
  │                                                                            │
  │  Possible Sources:                                                         │
  │    • AI/GPU market dominance                                               │
  │    • Superior earnings growth                                              │
  │    • Sector rotation into tech                                             │""")
    else:
        print(f"""  │  ❌ α < 0: NVDA generated NEGATIVE alpha                                   │
  │                                                                            │
  │  Interpretation:                                                           │
  │    • NVDA underperformed what its market exposure would predict            │
  │    • Even with high beta in a bull market, NVDA lagged expectations        │""")
    
    print("""  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘
""")

def print_strategy_evaluation():
    """Evaluate trading strategy scenarios."""
    print_section_header("TRADING STRATEGY EVALUATION", "🤖")
    
    print("""
  ╭────────────────────────────────────────────────────────────────────────────╮
  │  SCENARIO ANALYSIS: Two Trading Bots                                       │
  │                                                                            │
  │  Market (SPY) returned: +10%                                               │
  ╰────────────────────────────────────────────────────────────────────────────╯
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  SCENARIO A: Conservative Bot                                               │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  • Bot Return:    +15%                                                      │
  │  • Beta:          0.1                                                       │
  │  • Expected Return from Market: 0.1 x 10% = 1%                              │
  │  • Alpha:         15% - 1% = +14%  ✅                                        │
  │                                                                             │
  │  Verdict: EXCELLENT! Nearly all return is from alpha (skill).               │
  │           Low market correlation = good diversifier.                        │
  └─────────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  SCENARIO B: Aggressive Bot                                                 │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  • Bot Return:    +25%                                                      │
  │  • Beta:          2.5                                                       │
  │  • Expected Return from Market: 2.5 x 10% = 25%                             │
  │  • Alpha:         25% - 25% = 0%  ⚠️                                         │
  │                                                                             │
  │  Verdict: DECEPTIVE! Zero alpha despite impressive return.                  │
  │           This is just 2.5x leveraged S&P 500 exposure.                     │
  │           In a -10% market, this bot loses -25%!                            │
  └─────────────────────────────────────────────────────────────────────────────┘
  
  ╭────────────────────────────────────────────────────────────────────────────╮
  │  💡 KEY INSIGHT                                                            │
  │                                                                            │
  │  Scenario A is BETTER despite lower absolute returns because:              │
  │    1. It generates true alpha (skill-based returns)                        │
  │    2. It has low correlation to market (diversification benefit)           │
  │    3. It won't blow up in a market crash                                   │
  │                                                                            │
  │  Scenario B is DANGEROUS because:                                          │
  │    1. All returns come from leverage, not skill                            │
  │    2. High beta = high drawdown risk in bear markets                       │
  │    3. A hedge fund claiming "25% returns" with β=2.5 is misleading         │
  │                                                                            │
  │  "Don't confuse leverage for alpha."                                       │
  ╰────────────────────────────────────────────────────────────────────────────╯
""")

def print_rolling_beta_analysis(rolling_df):
    """Analyse the rolling beta results."""
    print_section_header("NON-STATIONARITY ANALYSIS (Rolling Beta)", "📉")
    
    betas = rolling_df['Rolling_Beta']
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  ROLLING BETA STATISTICS (6-month window)                                  │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  • Mean Beta:     {betas.mean():.3f}                                                    │
  │  • Min Beta:      {betas.min():.3f}                                                    │
  │  • Max Beta:      {betas.max():.3f}                                                    │
  │  • Std Dev:       {betas.std():.3f}                                                    │
  │  • Range:         {betas.max() - betas.min():.3f}                                                    │
  └────────────────────────────────────────────────────────────────────────────┘
  
  ╭────────────────────────────────────────────────────────────────────────────╮
  │  💡 NON-STATIONARITY INSIGHT                                               │
  │                                                                            │
  │  The rolling beta reveals that the NVDA-SPY relationship is NOT constant:  │
  │                                                                            │""")
    
    if betas.std() > 0.3:
        print(f"""  │  ⚠️  HIGH VARIABILITY: Beta ranges from {betas.min():.2f} to {betas.max():.2f}                       │
  │                                                                            │
  │  This means:                                                               │
  │    • A single "static" beta is misleading                                  │
  │    • Risk management must adapt to changing correlations                   │
  │    • During market stress, correlations often spike                        │""")
    else:
        print(f"""  │  ✅ MODERATE VARIABILITY: Beta relatively stable                           │
  │                                                                            │
  │  This suggests:                                                            │
  │    • The NVDA-SPY relationship is reasonably consistent                    │
  │    • Static beta is a reasonable approximation                             │""")
    
    print("""  │                                                                            │
  │  📊 Key Observations:                                                      │
  │    • Beta often INCREASES during market crashes (flight to correlation)   │
  │    • Beta may DECREASE when NVDA has idiosyncratic moves (earnings, etc.) │
  │    • This is why quants use "rolling" or "conditional" beta models        │
  ╰────────────────────────────────────────────────────────────────────────────╯
""")

def print_footer():
    """Print the project footer."""
    print("\n" + "═" * 80)
    print("""
  📝 METHODOLOGY NOTES:
  
  • CAPM Model: R_NVDA = α + β x R_SPY + ε
  • β estimated via OLS: β̂ = Cov(R_NVDA, R_SPY) / Var(R_SPY)
  • α estimated as: α̂ = R̄_NVDA - β̂ x R̄_SPY
  • Rolling beta uses 126-day window (~6 months)
  • Returns are simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
  
  📚 KEY CONCEPTS FROM THIS PROJECT:
  
  • Beta (β): Systematic risk - how much the asset moves with the market
  • Alpha (α): Idiosyncratic return - the "edge" independent of market
  • R²: How much of the asset's variance is explained by the market
  • Non-stationarity: Parameters drift over time (rolling beta reveals this)
  
  ⚠️  DISCLAIMER: Past performance does not guarantee future results.
      These analyses are for educational purposes only.
""")
    print("═" * 80)
    print("  End of Portfolio Alpha and Beta Analysis")
    print("═" * 80 + "\n")

#=================================================================
# MAIN EXECUTION
#=================================================================
def run_analysis_with_ibapi():
    """Main function to run the complete analysis using IB API."""
    print_header()

    print_section_header("CONNECTING TO INTERACTIVE BROKERS", "🔌")
    print(" Attempting to connect to TWS/IB Gateway...")
    print(" (Ensure TWS or IB Gateway is running on localhost:7497)")
    print()

    app = IBDataFetcher()
    try:
        app.connect("127.0.0.1", 7497, clientId=3)
    except Exception as e:
        print(f" ❌ Connection failed: {e}")
        return None
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    if not app.connection_established.wait(timeout=10):
        print("  ❌ Timeout waiting for connection")
        app.disconnect()
        return None
    
    print("  ✅ Connected successfully!")

    # Fetch NVDA data
    print_section_header("FETCHING HISTORICAL DATA", "📥")
    
    print("  Requesting 5 years of NVDA daily data...")
    nvda_contract = create_contract("NVDA", primary_exchange="NASDAQ")
    nvda_df = fetch_historical_data(app, nvda_contract, duration="5 Y")
    
    if nvda_df.empty:
        print("  ❌ Failed to fetch NVDA data")
        app.disconnect()
        return None
    print(f"  ✅ Received {len(nvda_df)} NVDA daily bars")
    
    time.sleep(1)  # Rate limiting
    
    print("  Requesting 5 years of SPY daily data...")
    spy_contract = create_contract("SPY", primary_exchange="ARCA")
    spy_df = fetch_historical_data(app, spy_contract, duration="5 Y")
    
    if spy_df.empty:
        print("  ❌ Failed to fetch SPY data")
        app.disconnect()
        return None
    print(f"  ✅ Received {len(spy_df)} SPY daily bars")
    
    app.disconnect()
    time.sleep(1)
    
    return run_analysis(nvda_df, spy_df)            

def run_analysis(nvda_df, spy_df):
    """Run the complete CAPM analysis."""
    
    # Calculate returns
    print("\n  Calculating daily returns...")
    nvda_df = calculate_returns(nvda_df)
    spy_df = calculate_returns(spy_df)
    
    # Merge datasets
    print("  Merging datasets on common dates...")
    merged_df = merge_datasets(nvda_df, spy_df)
    print(f"  ✅ Merged dataset: {len(merged_df)} observations")
    
    # Display data summary
    print_data_summary(nvda_df, spy_df, merged_df)
    
    # Run regression
    print_section_header("RUNNING REGRESSION ANALYSIS", "🔧")
    
    y = merged_df['NVDA_Return'].values
    x = merged_df['SPY_Return'].values
    
    print("  Estimating β and α using OLS...")
    beta, alpha = calculate_beta_alpha(y, x)
    
    # Calculate predicted values and metrics
    y_pred = alpha + beta * x
    metrics = calculate_regression_metrics(y, y_pred)
    
    print(f"  ✅ β = {beta:.4f}, α = {alpha*252*100:.2f}% (annualised)")
    
    # Calculate rolling beta
    print("  Calculating rolling beta (6-month window)...")
    rolling_df = calculate_rolling_beta(merged_df)
    print(f"  ✅ Rolling beta calculated for {len(rolling_df)} periods")
    
    # Display results
    print_regression_results(beta, alpha, metrics)
    print_beta_interpretation(beta, metrics)
    print_alpha_interpretation(alpha, beta)
    print_strategy_evaluation()
    print_rolling_beta_analysis(rolling_df)
    
    # Create visualisations
    print_section_header("GENERATING VISUALISATIONS", "📊")
    print("  Creating regression scatter plot...")
    fig1 = create_regression_plot(merged_df, beta, alpha, metrics)
    
    print("  Creating rolling beta plot...")
    fig2 = create_rolling_beta_plot(rolling_df)
    
    print("  Creating cumulative returns comparison...")
    fig3 = create_cumulative_returns_plot(merged_df)
    
    print("  ✅ All plots generated!")
    print("\n  📊 Displaying plots... (close plot windows to continue)")
    
    # Footer
    print_footer()
    
    # Show all plots
    plt.show()
    
    return {
        'merged_data': merged_df,
        'beta': beta,
        'alpha': alpha,
        'metrics': metrics,
        'rolling_beta': rolling_df
    }

#==========================================================
# DEMO MODE
#==========================================================
def generate_sample_data():
    """Generate realistic sample NVDA and SPY data."""
    np.random.seed(42)
    
    start_date = datetime(2021, 1, 4)
    n_days = 1260
    
    dates = pd.bdate_range(start=start_date, periods=n_days)
    
    # Generate SPY returns (market)
    spy_returns = np.random.normal(0.0004, 0.012, n_days)  # ~10% annual, 19% vol
    
    # Generate NVDA returns with beta exposure + alpha + idiosyncratic
    beta_true = 1.8
    alpha_true = 0.0008  # ~20% annual alpha
    idiosyncratic_vol = 0.02
    
    nvda_returns = (alpha_true + 
                    beta_true * spy_returns + 
                    np.random.normal(0, idiosyncratic_vol, n_days))
    
    # Add some regime changes
    crash_periods = [(200, 230), (700, 750)]  # Simulate crashes
    for start, end in crash_periods:
        spy_returns[start:end] *= 2.5  # Higher vol
        nvda_returns[start:end] *= 3  # Even higher vol (beta increases in crashes)
    
    # Calculate prices
    spy_prices = 400 * np.exp(np.cumsum(spy_returns))
    nvda_prices = 200 * np.exp(np.cumsum(nvda_returns))
    
    # Create DataFrames
    def create_ohlcv(dates, prices):
        data = []
        for date, close in zip(dates, prices):
            daily_range = close * np.random.uniform(0.005, 0.02)
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            open_price = np.random.uniform(low, high)
            volume = int(np.random.uniform(20_000_000, 60_000_000))
            data.append({
                'Date': date, 'Open': open_price, 'High': high,
                'Low': low, 'Close': close, 'Volume': volume
            })
        return pd.DataFrame(data)
    
    return create_ohlcv(dates, nvda_prices), create_ohlcv(dates, spy_prices)


def run_demo_mode():
    """Run analysis with sample data."""
    print_header()
    
    print_section_header("DEMO MODE", "🎮")
    print("""
  ⚠️  Running in DEMO MODE with simulated NVDA and SPY data.
  
  The simulation includes:
    • NVDA with β ≈ 1.8 (high beta tech stock)
    • Positive alpha component
    • Regime changes (crash periods with higher correlations)
  
  To use real data via Interactive Brokers:
    1. Ensure TWS or IB Gateway is running
    2. Enable API connections on port 7497
    3. Call run_analysis_with_ibapi() instead
""")
    
    print("  Generating 5 years of simulated data...")
    nvda_df, spy_df = generate_sample_data()
    print(f"  ✅ Generated {len(nvda_df)} daily bars for each asset")
    
    return run_analysis(nvda_df, spy_df)

#==========================================================
# ENTRY POINT
#==========================================================

if __name__ == "__main__":
    try:
        from ibapi.client import EClient
        print("\n  Attempting to connect to Interactive Brokers...")
        result = run_analysis_with_ibapi()
        
        if result is None:
            print("\n  Falling back to demo mode...\n")
            result = run_demo_mode()
            
    except ImportError:
        print("\n  ⚠️ IBAPI not installed. Running demo mode.")
        print("  To install: pip install ibapi")
        result = run_demo_mode()
    except Exception as e:
        print(f"\n  ⚠️ Error connecting to IB: {e}")
        print("  Running demo mode instead...\n")
        result = run_demo_mode()


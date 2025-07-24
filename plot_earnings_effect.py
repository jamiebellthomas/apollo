import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime, timedelta

# Enable LaTeX-like styling
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Define companies and earnings dates
examples = {
    'AAPL': '2023-05-04',  # Q2 2023 earnings
    'META': '2022-10-26',  # Q3 2022 earnings
}

def plot_earnings_reaction(ticker, earnings_date, quarter:str, days_before=10, days_after=10):
    earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')

    start = earnings_dt - timedelta(days=days_before + 3)
    end = earnings_dt + timedelta(days=days_after + 3)

    stock = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    stock = stock.loc[~stock.index.duplicated(keep='first')]

    stock_window = stock[(stock.index >= earnings_dt - timedelta(days=days_before)) &
                         (stock.index <= earnings_dt + timedelta(days=days_after))]

    plt.figure(figsize=(10, 5))
    plt.plot(stock_window.index, stock_window['Close'], marker='o', linestyle='-')
    plt.axvline(earnings_dt, color='red', linestyle='--', label=r'\textbf{Earnings Release}')
    plt.title(rf'\textbf{{{ticker} Stock Price Reaction to Earnings Report Released in {quarter} of {earnings_date[:4]}}}', fontsize=14)
    plt.xlabel(r'\textbf{Date}', fontsize=14)
    plt.ylabel(r'\textbf{Stock Price}', fontsize=14)
    plt.legend(fontsize=14)
    plt.xticks(rotation=0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Plots/earnings_reaction/earnings_reaction_{ticker}_{quarter}.png', dpi=300)
    plt.close()
    # Increase legend text size
    


# Generate the plots
plot_earnings_reaction('AAPL', examples['AAPL'], quarter='Q2')
plot_earnings_reaction('META', examples['META'], quarter='Q3')

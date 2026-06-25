"""Fetch raw market data from Yahoo Finance for the stress regime project."""

from datetime import date

import pandas as pd
import yfinance as yf

SECTOR_ETFS = ["XLF", "XLK", "XLE", "XLU", "XLY", "XLP", "XLV", "XLI"]
RISK_FACTORS = ["^VIX", "^TNX", "^IRX"]
TICKERS = SECTOR_ETFS + RISK_FACTORS

START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date, one column per ticker. Tickers that
    fail to download are skipped with a printed warning instead of breaking
    the whole run, since Yahoo Finance occasionally drops a ticker.
    """
    frames = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            print(f"no data returned for {ticker}, skipping")
            continue
        frames[ticker] = data["Adj Close"]

    prices = pd.DataFrame(frames)
    prices = prices.dropna(how="all")
    return prices


def save_raw(prices: pd.DataFrame, path: str) -> None:
    prices.to_csv(path)


if __name__ == "__main__":
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    save_raw(prices, "data/raw/prices.csv")
    print(f"saved {prices.shape[0]} rows, {prices.shape[1]} tickers")

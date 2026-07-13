"""Feature engineering for the market stress classification task."""

import numpy as np
import pandas as pd

REALIZED_VOL_WINDOW = 20
MOMENTUM_WINDOW = 10
STRESS_QUANTILE = 0.75
STRESS_MIN_HISTORY = 252
TRADING_DAYS_PER_YEAR = 252
FEATURE_COLUMNS = ["portfolio_vol", "vix", "yield_spread", "momentum"]


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def realized_volatility(returns: pd.DataFrame, window: int = REALIZED_VOL_WINDOW) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def yield_spread(tnx: pd.Series, irx: pd.Series) -> pd.Series:
    """10 year minus 13 week treasury yield. Negative values mean an inverted curve."""
    return tnx - irx


def momentum(price_series: pd.Series, window: int = MOMENTUM_WINDOW) -> pd.Series:
    return price_series.pct_change(periods=window)


def label_stress_regime(
    portfolio_vol: pd.Series,
    quantile: float = STRESS_QUANTILE,
    min_history: int = STRESS_MIN_HISTORY,
) -> pd.Series:
    """Label stress relative to volatility observed before each date.

    The expanding threshold is shifted by one day, so neither the current
    observation nor future test data can influence the label boundary.
    """
    threshold = (
        portfolio_vol.shift(1)
        .expanding(min_periods=min_history)
        .quantile(quantile)
    )
    labels = (portfolio_vol > threshold).astype("Int64")
    return labels.where(threshold.notna())


def build_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Combine returns, volatility, yield spread and momentum into one table.

    Expects prices to include the sector ETFs plus ^VIX, ^TNX and ^IRX
    columns, as produced by data_fetch.fetch_prices.

    Features are lagged by one day relative to the stress label. The stress
    threshold is also based only on observations available before the label
    date. Together these choices make the split suitable for one-day-ahead
    evaluation without using the held-out period to define its labels.
    """
    sector_cols = [c for c in prices.columns if not c.startswith("^")]

    sector_returns = compute_returns(prices[sector_cols])
    portfolio_returns = sector_returns.mean(axis=1)
    portfolio_price = portfolio_returns.add(1).cumprod()
    portfolio_vol = realized_volatility(portfolio_returns.to_frame("portfolio"))["portfolio"]

    features = pd.DataFrame(index=prices.index)
    features["portfolio_vol"] = portfolio_vol.shift(1)
    features["vix"] = prices["^VIX"].shift(1)
    features["yield_spread"] = yield_spread(prices["^TNX"], prices["^IRX"]).shift(1)
    features["momentum"] = momentum(portfolio_price).shift(1)
    features["stress"] = label_stress_regime(portfolio_vol)

    features = features.dropna()
    features["stress"] = features["stress"].astype(int)
    return features

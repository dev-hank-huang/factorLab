from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('LAVD')
class LAVDCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Step 1: Retrieve intraday data
        high = data.get("price:high")
        low = data.get("price:low")
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)

        if resample.upper() != "D":
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()

        # Step 2: Calculate daily returns
        returns = close.pct_change().fillna(0)

        # Step 3: Calculate realized volatility
        realized_vol = returns.rolling(window=21).std() * np.sqrt(252)

        # Step 4: Calculate liquidity-adjusted volatility
        vwap_volume = volume.rolling(window=21).mean()
        lav = realized_vol / np.sqrt(vwap_volume)

        # Step 5: Compute the volatility disparity
        lavd = lav / realized_vol - 1

        # Step 6: Normalize the scores cross-sectionally
        lavd_zscore = (lavd - lavd.mean(axis=1)) / lavd.std(axis=1)

        # Wrap results into CustomDataFrame
        return CustomDataFrame(lavd_zscore)


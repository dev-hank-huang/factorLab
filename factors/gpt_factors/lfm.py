from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
from dataframe import CustomDataFrame
import numpy as np

@FactorRegistry.register('LFM')
class LFMCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Step 1: Retrieve and preprocess core data
        close = data.get("price:close")
        volume = data.get("price:volume")
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()

        log_returns = np.log(close).diff()
        
        # Step 2: Calculate market volume for normalization
        total_market_volume = volume.sum(axis=1)
        volume_ratio = volume.div(total_market_volume, axis=0)
        
        # Step 3: Compute rolling correlation (Liquidity Flow Momentum)
        dfs = {}
        window = 5  # Default window for correlation
        
        for col in log_returns.columns:
            price_diff = log_returns[col]
            volume_ratio_col = volume_ratio[col]
            
            # Handle missing data gracefully
            if price_diff.isnull().all() or volume_ratio_col.isnull().all():
                result = pd.Series(index=close.index, dtype=float)
            else:
                result = price_diff.rolling(window=window).corr(volume_ratio_col)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][col] = result
            
        # Step 4: Return formatted result as CustomDataFrame
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])

from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('RLTF')
class RLTFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required price data
        close = data.get("price:close")
        high = data.get("price:high")
        low = data.get("price:low")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        close.index = pd.to_datetime(close.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        volume.index = pd.to_datetime(volume.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
        
        dfs = {}
        for key in close.columns:
            # Extract individual series
            c = close[key].ffill()
            h = high[key].ffill()
            l = low[key].ffill()
            v = volume[key].ffill()
            
            # Calculate Range Evolution Metric
            current_range = h - l
            historical_range = current_range.rolling(window=20).mean()
            
            # Calculate volatilities
            returns = c.pct_change()
            current_vol = returns.rolling(window=10).std()
            historical_vol = returns.rolling(window=60).std()
            
            rem = (current_range / historical_range.shift(1)) * (historical_vol / current_vol)
            
            # Calculate Liquidity Transition Signal
            volume_ma = v.rolling(window=20).mean()
            range_ma = historical_range
            
            lts = (v / volume_ma) * (current_range / range_ma)
            
            # Calculate Range Breakout Probability
            price_location = (c - l) / current_range.replace(0, np.nan)
            median_location = price_location.rolling(window=60).median()
            
            rbp = price_location - median_location
            
            # Combine components with dynamic weights
            volatility_regime = current_vol / historical_vol
            volume_regime = v / volume_ma
            
            # Adjust weights based on market conditions
            rem_weight = 0.4 * (1 + volatility_regime.clip(0, 1))
            lts_weight = 0.35 * (1 + volume_regime.clip(0, 1))
            rbp_weight = 1 - rem_weight - lts_weight
            
            # Calculate final factor value
            result = (rem_weight * rem.fillna(0) +
                     lts_weight * lts.fillna(0) +
                     rbp_weight * rbp.fillna(0))
            
            # Apply sigmoid transformation to control outliers
            result = 2 / (1 + np.exp(-result)) - 1
            
            # Standardize the result
            result = (result - result.rolling(window=252).mean()) / result.rolling(window=252).std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
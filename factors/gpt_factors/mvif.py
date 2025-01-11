from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('MVIF')
class MVIFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required price data
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            
        dfs = {}
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate Momentum Signal
            returns_5 = price.pct_change(5)
            returns_20 = price.pct_change(20)
            
            # Calculate volatility adjustments using exponential weights
            vol_5 = returns_5.ewm(span=5).std()
            vol_20 = returns_20.ewm(span=20).std()
            
            # Momentum component with volatility adjustment
            ms = (returns_5 / vol_5) + (returns_20 / vol_20)
            
            # Volume Force calculation
            vol_ema = vol.ewm(span=20).mean()
            vol_ratio = vol / vol_ema
            
            # Apply exponential decay to recent volume changes
            decay_factor = 0.1
            vol_changes = vol_ratio.diff(5)
            vol_force = vol_changes * np.exp(-decay_factor * np.arange(len(vol_changes)))
            
            # Dynamic Volatility Weighting
            rolling_vol = returns_20.rolling(window=20).std()
            vol_percentile = rolling_vol.rolling(window=252).rank(pct=True)
            
            # Combine components with adaptive weights
            base_momentum_weight = 0.5
            base_volume_weight = 0.3
            base_vol_weight = 0.2
            
            # Adjust weights based on market conditions
            momentum_weight = base_momentum_weight * (1 + vol_percentile)
            volume_weight = base_volume_weight * (1 + vol_ratio.rolling(window=20).rank(pct=True))
            vol_weight = 1 - momentum_weight - volume_weight
            
            # Calculate final factor value
            result = (momentum_weight * ms.fillna(0) +
                     volume_weight * vol_force.fillna(0) +
                     vol_weight * vol_percentile.fillna(0))
            
            # Standardize the result
            result = (result - result.rolling(window=252).mean()) / result.rolling(window=252).std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
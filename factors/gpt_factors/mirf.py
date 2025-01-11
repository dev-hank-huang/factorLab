from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('MIRF')
class MIRFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required data
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
            # Extract price and volume series
            c = close[key].ffill()
            h = high[key].ffill()
            l = low[key].ffill()
            v = volume[key].ffill()
            
            # Calculate Price Recovery Efficiency
            price_range = h - l
            relative_close = (c - l) / price_range.replace(0, np.nan)
            volume_ratio = v / v.rolling(window=20).mean()
            
            pre = relative_close * (1 + volume_ratio)
            
            # Calculate Intraday Stability Metric
            returns = c.pct_change()
            historical_vol = returns.rolling(window=60).std()
            sqrt_volume = np.sqrt(v)
            
            ism = (price_range / (sqrt_volume * historical_vol)).replace([np.inf, -np.inf], np.nan)
            
            # Calculate Trading Pressure Absorption
            def rolling_correlation(x, y, window=20):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            price_changes = returns.abs()
            volume_shocks = (v - v.rolling(window=20).mean()) / v.rolling(window=20).std()
            
            pressure_correlation = rolling_correlation(price_changes, volume_shocks)
            range_ratio = price_range / price_range.rolling(window=20).mean()
            
            tpa = pressure_correlation * (1 + range_ratio)
            
            # Dynamic weight calculation based on market conditions
            volatility_state = historical_vol / historical_vol.rolling(window=252).mean()
            volume_state = volume_ratio.rolling(window=20).mean()
            
            # Adjust component weights based on market conditions
            pre_weight = 0.4 * (1 + volatility_state.clip(0, 1))
            ism_weight = 0.35 * (1 + volume_state.clip(0, 1))
            tpa_weight = 1 - pre_weight - ism_weight
            
            # Calculate final factor value
            result = (pre_weight * pre.fillna(0) +
                     ism_weight * (-ism).fillna(0) +  # Negative because lower ISM indicates higher stability
                     tpa_weight * (-tpa).fillna(0))   # Negative because lower TPA indicates better resilience
            
            # Apply sigmoid transformation for outlier control
            result = 2 / (1 + np.exp(-result)) - 1
            
            # Standardize with expanding window
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
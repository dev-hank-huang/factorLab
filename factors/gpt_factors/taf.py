from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('TAF')
class TAFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Fetch required price data
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
            # Extract individual stock data
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate Relative Price Oscillator (RPO)
            ma_10 = price.rolling(window=10).mean()
            ma_20 = price.rolling(window=20).mean()
            trend_strength = abs(ma_10.pct_change(5))
            rpo = (ma_20 - ma_10) / ma_20.where(ma_20 != 0, np.nan) * (1 + trend_strength)
            
            # Calculate Volume Acceleration (VA)
            va = vol.pct_change(periods=3)
            vol_strength = abs(va) / va.rolling(window=20).std()
            va = va * (1 + vol_strength)
            
            # Calculate Historical Price Movement (HPM)
            price_change = price.pct_change(10)
            price_std = price.rolling(window=10).std()
            hpm = price_change / price_std.where(price_std != 0, np.nan)
            
            # Calculate market condition indicators
            trend_adjustment = trend_strength.rolling(window=10).mean()
            vol_adjustment = vol_strength.rolling(window=10).mean()
            
            # Normalize components
            rpo_norm = (rpo - rpo.rolling(window=20).mean()) / rpo.rolling(window=20).std()
            va_norm = (va - va.rolling(window=20).mean()) / va.rolling(window=20).std()
            hpm_norm = (hpm - hpm.rolling(window=20).mean()) / hpm_norm.rolling(window=20).std()
            
            # Base weights
            base_rpo_weight = 0.4
            base_va_weight = 0.35
            base_hpm_weight = 0.25
            
            # Calculate final weights
            rpo_weight = base_rpo_weight * (1 + 0.2 * trend_adjustment)
            va_weight = base_va_weight * (1 + 0.2 * vol_adjustment)
            hpm_weight = 1 - rpo_weight - va_weight
            
            # Generate trend signals
            trend_signal = ((rpo_norm > 0) & (va_norm > 0) & (hpm_norm > 0)).astype(float)
            
            # Combine components into final factor
            result = (rpo_weight * rpo_norm + 
                     va_weight * va_norm + 
                     hpm_weight * hpm_norm) * trend_signal
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('MIMF')
class MIMFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Fetch required price data
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
            # Extract individual stock data
            price = close[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate Price Momentum Score (PMS)
            price_mean = price.rolling(window=5).mean()
            price_std = price.rolling(window=5).std()
            pms = (price - price_mean) / price_std.where(price_std != 0, np.nan)
            
            # Calculate Average True Range (ATR) for VIR
            tr = pd.DataFrame({
                'h-l': high_price - low_price,
                'h-pc': abs(high_price - price.shift(1)),
                'l-pc': abs(low_price - price.shift(1))
            }).max(axis=1)
            atr = tr.rolling(window=10).mean()
            
            # Calculate Volume Impact Ratio (VIR)
            vol_mean = vol.rolling(window=10).mean()
            daily_range = high_price - low_price
            vir = (vol * daily_range) / (vol_mean * atr).where(vol_mean * atr != 0, np.nan)
            
            # Calculate Range Expansion Velocity (REV)
            high_max = high_price.rolling(window=10).max()
            low_min = low_price.rolling(window=10).min()
            rev = daily_range / (high_max - low_min).where(high_max - low_min != 0, np.nan)
            
            # Normalize components
            pms_norm = (pms - pms.rolling(window=20).mean()) / pms.rolling(window=20).std()
            vir_norm = (vir - vir.rolling(window=20).mean()) / vir.rolling(window=20).std()
            rev_norm = (rev - rev.rolling(window=20).mean()) / rev.rolling(window=20).std()
            
            # Calculate market stress indicator for dynamic weighting
            volatility_ratio = price.pct_change().rolling(window=20).std() / \
                             price.pct_change().rolling(window=60).std()
            volume_ratio = vol / vol.rolling(window=20).mean()
            
            # Dynamic weight calculation
            base_weights = np.array([0.4, 0.35, 0.25])  # Base weights for PMS, VIR, REV
            stress_multiplier = np.clip(volatility_ratio * volume_ratio, 0.8, 1.2)
            
            # Apply stress multiplier to weights
            w1 = base_weights[0] * stress_multiplier
            w2 = base_weights[1] * (2 - stress_multiplier)
            w3 = 1 - (w1 + w2)  # Ensure weights sum to 1
            
            # Combine components into final factor
            result = (w1 * pms_norm + 
                     w2 * vir_norm + 
                     w3 * rev_norm)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
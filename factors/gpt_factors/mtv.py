from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('MTV')
class MTVCalculator(FactorCalculator):
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
            
            # Calculate Momentum Score (MS)
            ms = price.pct_change(periods=5)
            
            # Calculate Volume Spike (VS)
            vs = vol.pct_change(periods=3)
            vol_std = vol.rolling(window=20).std()
            vs_normalized = vs / vol_std.where(vol_std != 0, np.nan)
            
            # Calculate Trading Range Fluctuation (TRF)
            daily_range = high_price - low_price
            tr = pd.DataFrame({
                'h-l': high_price - low_price,
                'h-pc': abs(high_price - price.shift(1)),
                'l-pc': abs(low_price - price.shift(1))
            }).max(axis=1)
            atr = tr.rolling(window=10).mean()
            trf = daily_range / atr.where(atr != 0, np.nan)
            
            # Calculate market condition indicators
            volatility = price.pct_change().rolling(window=20).std()
            vol_ratio = vol / vol.rolling(window=20).mean()
            
            # Normalize components
            ms_norm = (ms - ms.rolling(window=20).mean()) / ms.rolling(window=20).std()
            vs_norm = (vs_normalized - vs_normalized.rolling(window=20).mean()) / vs_normalized.rolling(window=20).std()
            trf_norm = (trf - trf.rolling(window=20).mean()) / trf.rolling(window=20).std()
            
            # Generate confirmation signals
            momentum_signal = (ms_norm > 0).astype(float)
            volume_signal = (vs_norm > 0.5).astype(float)
            range_signal = (trf_norm > 0).astype(float)
            
            # Calculate signal strength
            signal_strength = momentum_signal * volume_signal * range_signal
            
            # Combine components into final factor
            result = (0.4 * ms_norm + 
                     0.35 * vs_norm + 
                     0.25 * trf_norm) * signal_strength
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
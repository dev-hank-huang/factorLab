from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('TREF')
class TREFCalculator(FactorCalculator):
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
            
            # Calculate daily trading range
            daily_range = high_price - low_price
            range_ema = daily_range.ewm(span=20).mean()
            
            # Calculate Trading Range Evolution (TRE)
            range_change = daily_range.pct_change()
            range_acceleration = range_change.diff()
            tre = (daily_range / range_ema.where(range_ema != 0, np.nan)) * (1 + range_acceleration)
            
            # Calculate Volume Range Correlation (VRC)
            vol_change = vol.pct_change()
            vrc = pd.Series(
                [range_change.iloc[i-10:i].corr(vol_change.iloc[i-10:i]) 
                 if i >= 10 else np.nan 
                 for i in range(len(range_change))]
            )
            
            # Calculate Range Momentum Score (RMS)
            range_ma5 = daily_range.rolling(window=5).mean()
            range_ma20 = daily_range.rolling(window=20).mean()
            rms = (range_ma5 - range_ma20) / range_ma20.where(range_ma20 != 0, np.nan)
            
            # Calculate market condition indicators
            volatility = price.pct_change().rolling(window=20).std()
            vol_ratio = vol / vol.rolling(window=20).mean()
            
            # Normalize components
            tre_norm = (tre - tre.rolling(window=20).mean()) / tre.rolling(window=20).std()
            vrc_norm = (vrc - vrc.rolling(window=20).mean()) / vrc.rolling(window=20).std()
            rms_norm = (rms - rms.rolling(window=20).mean()) / rms.rolling(window=20).std()
            
            # Calculate adaptive weights based on market conditions
            volatility_regime = volatility / volatility.rolling(window=60).mean()
            volume_regime = vol_ratio.rolling(window=10).mean()
            
            # Base weights
            w1 = 0.4  # TRE weight
            w2 = 0.35 # VRC weight
            w3 = 0.25 # RMS weight
            
            # Adjust weights based on market conditions
            regime_multiplier = np.clip(volatility_regime * volume_regime, 0.8, 1.2)
            
            w1_adj = w1 * regime_multiplier
            w2_adj = w2 * (2 - regime_multiplier)
            w3_adj = 1 - (w1_adj + w2_adj)
            
            # Generate range expansion signals
            range_signal = ((tre_norm > 0) & (vrc_norm > 0)).astype(float)
            
            # Combine components into final factor
            result = (w1_adj * tre_norm + 
                     w2_adj * vrc_norm + 
                     w3_adj * rms_norm) * range_signal
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
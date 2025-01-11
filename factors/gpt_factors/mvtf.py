from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('MVTF')
class MVTFCalculator(FactorCalculator):
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
            
            # Calculate Volume-Momentum Interaction (VMI)
            # Short-term (7 days) and long-term (14 days) momentum
            mom_7 = price - price.shift(7)
            mom_14 = price - price.shift(14)
            
            # Volume standard deviations
            vol_std_7 = vol.rolling(window=7).std()
            vol_std_14 = vol.rolling(window=14).std()
            
            vmi = (mom_7/vol_std_14) * (mom_14/vol_std_7)
            
            # Calculate Volume Time Decay (VTD)
            vol_diff = vol - vol.shift(3)
            decay_factors = np.exp([-0.5 * i for i in range(3)])
            vtd = pd.Series(0, index=vol.index)
            
            for i in range(3):
                vtd += vol_diff.shift(i) * decay_factors[i]
            
            # Calculate Trading Range Persistence (TRP)
            daily_range = high_price - low_price
            ema_range = daily_range.ewm(span=10).mean()
            atr = daily_range.rolling(window=14).mean()
            
            trp = (daily_range - ema_range) / atr.where(atr != 0, np.nan)
            
            # Calculate market condition indicators
            volatility = price.pct_change().rolling(window=20).std()
            vol_ratio = vol / vol.rolling(window=20).mean()
            
            # Normalize components
            vmi_norm = (vmi - vmi.rolling(window=20).mean()) / vmi.rolling(window=20).std()
            vtd_norm = (vtd - vtd.rolling(window=20).mean()) / vtd.rolling(window=20).std()
            trp_norm = (trp - trp.rolling(window=20).mean()) / trp.rolling(window=20).std()
            
            # Calculate adaptive weights based on market conditions
            volatility_regime = volatility / volatility.rolling(window=60).mean()
            volume_regime = vol_ratio.rolling(window=10).mean()
            
            # Base weights
            w1 = 0.4  # VMI weight
            w2 = 0.35 # VTD weight
            w3 = 0.25 # TRP weight
            
            # Adjust weights based on market conditions
            regime_multiplier = np.clip(volatility_regime * volume_regime, 0.8, 1.2)
            
            w1_adj = w1 * regime_multiplier
            w2_adj = w2 * (2 - regime_multiplier)
            w3_adj = 1 - (w1_adj + w2_adj)
            
            # Generate confirmation signals
            momentum_signal = (vmi_norm > 0).astype(float)
            volume_signal = (vtd_norm > 0).astype(float)
            
            # Combine components into final factor
            result = (w1_adj * vmi_norm + 
                     w2_adj * vtd_norm + 
                     w3_adj * trp_norm) * momentum_signal * volume_signal
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
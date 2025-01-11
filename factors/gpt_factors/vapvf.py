from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('VAPVF')
class VAPVFCalculator(FactorCalculator):
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
            
            # Calculate Price-Volume Trend (PVT)
            price_change = price.pct_change()
            volume_change = vol.pct_change()
            pvt = ((price_change * volume_change)).cumsum()
            
            # Calculate Normalized Intraday Volatility (NIV)
            daily_range = high_price - low_price
            ma20_price = price.rolling(window=20).mean()
            niv = daily_range / ma20_price.where(ma20_price != 0, np.nan)
            
            # Calculate Trading Range Differential (TRD)
            range_ma5 = daily_range.rolling(window=5).mean()
            trd = (daily_range - range_ma5) / range_ma5.where(range_ma5 != 0, np.nan)
            
            # Calculate market condition indicators
            volatility = price.pct_change().rolling(window=20).std()
            vol_ratio = vol / vol.rolling(window=20).mean()
            
            # Normalize components
            pvt_norm = (pvt - pvt.rolling(window=20).mean()) / pvt.rolling(window=20).std()
            niv_norm = (niv - niv.rolling(window=20).mean()) / niv.rolling(window=20).std()
            trd_norm = (trd - trd.rolling(window=20).mean()) / trd.rolling(window=20).std()
            
            # Calculate adaptive weights based on market conditions
            volatility_regime = volatility / volatility.rolling(window=60).mean()
            volume_regime = vol_ratio.rolling(window=10).mean()
            
            # Adjust weights based on market conditions
            regime_multiplier = np.clip(volatility_regime * volume_regime, 0.8, 1.2)
            
            w1 = 0.4 * regime_multiplier        # PVT weight
            w2 = 0.3 * (2 - regime_multiplier)  # NIV weight
            w3 = 1 - (w1 + w2)                  # TRD weight
            
            # Generate confirmation signals
            trend_signal = (pvt_norm > 0).astype(float)
            volatility_signal = (niv_norm < 2).astype(float)  # Filter extreme volatility
            
            # Combine components into final factor
            result = (w1 * pvt_norm + 
                     w2 * niv_norm + 
                     w3 * trd_norm) * trend_signal * volatility_signal
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
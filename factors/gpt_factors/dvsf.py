from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('DVSF')
class DVSFCalculator(FactorCalculator):
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
            price = close[key].ffill()
            vol = volume[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            
            # Calculate Volume Pressure Score (VPS)
            vol_ma20 = vol.rolling(window=20).mean()
            vol_std20 = vol.rolling(window=20).std()
            price_ma10 = price.rolling(window=10).mean()
            price_std10 = price.rolling(window=10).std()
            
            vps = ((vol - vol_ma20) / vol_std20.where(vol_std20 != 0, 1)) * \
                  ((price - price_ma10) / price_std10.where(price_std10 != 0, 1))
            
            # Calculate Price Acceleration Indicator (PAI)
            price_diff = price.diff()
            price_acc = price_diff - price_diff.shift(1)
            daily_range = (high_price - low_price) / price
            pai = price_acc / daily_range.where(daily_range != 0, 1)
            
            # Calculate Sentiment Oscillator (SO)
            vps_pai = vps * pai
            ema_5 = vps_pai.ewm(span=5, adjust=False).mean()
            ema_20 = abs(vps_pai).ewm(span=20, adjust=False).mean()
            so = ema_5 / ema_20.where(ema_20 != 0, 1)
            
            # Calculate final DVSF score
            vol_ma60 = vol.rolling(window=60).mean()
            volume_ratio = np.log1p(vol / vol_ma60.where(vol_ma60 != 0, 1))
            dvsf = so * (1 + volume_ratio)
            
            # Store results
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = dvsf
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('DVMF')
class DVMFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required price data
        close = data.get("price:close")
        open_ = data.get("price:open")
        high = data.get("price:high")
        low = data.get("price:low")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        close.index = pd.to_datetime(close.index)
        open_.index = pd.to_datetime(open_.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        volume.index = pd.to_datetime(volume.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            open_ = open_.resample(resample).first()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
        
        dfs = {}
        for key in close.columns:
            # Get individual stock data
            price = close[key].ffill()
            open_price = open_[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate VPC (Volume-Weighted Price Change)
            avg_volume = vol.rolling(window=20).mean()
            vpc = ((price - open_price) * (vol / avg_volume))
            
            # Calculate IRR (Intraday Range Ratio)
            mid_price = (high_price + low_price) / 2
            irr = (high_price - low_price) / mid_price.where(mid_price != 0, np.nan)
            
            # Calculate VA (Volume Acceleration)
            vol_mean = vol.rolling(window=5).mean()
            vol_std = vol.rolling(window=5).std()
            va = (vol - vol_mean) / vol_std.where(vol_std != 0, np.nan)
            
            # Dynamic weight calculation based on market volatility
            volatility = price.pct_change().rolling(window=20).std()
            vol_weight = volatility / volatility.rolling(window=60).mean()
            vol_weight = vol_weight.clip(0.5, 1.5)
            
            # Calculate component weights
            w1 = 0.4 * vol_weight
            w2 = 0.3 * (2 - vol_weight)
            w3 = 0.3 * vol_weight
            
            # Normalize components
            vpc_norm = (vpc - vpc.rolling(window=20).mean()) / vpc.rolling(window=20).std()
            irr_norm = (irr - irr.rolling(window=20).mean()) / irr.rolling(window=20).std()
            va_norm = (va - va.rolling(window=20).mean()) / va.rolling(window=20).std()
            
            # Combine components into final factor
            result = (w1 * vpc_norm + 
                     w2 * irr_norm + 
                     w3 * va_norm)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
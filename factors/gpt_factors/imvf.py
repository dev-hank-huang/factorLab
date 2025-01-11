from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('IMVF')
class IMVFCalculator(FactorCalculator):
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
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            open_ = open_.resample(resample).first()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
            
        dfs = {}
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            open_price = open_[key].ffill()
            
            # Calculate IPE
            daily_range = high_price - low_price
            ipe = (price - open_price) / daily_range.where(daily_range != 0, 1)
            
            # Calculate VVR
            vol_ma20 = vol.rolling(window=20).mean()
            price_std10 = price.rolling(window=10).std()
            price_std60 = price.rolling(window=60).std()
            vvr = (vol / vol_ma20.where(vol_ma20 != 0, 1)) / (
                price_std10 / price_std60.where(price_std60 != 0, 1)
            )
            
            # Calculate MQS
            price_ma10 = price.rolling(window=10).mean()
            vol_ma10 = vol.rolling(window=10).mean()
            mqs = (price - price_ma10) * (vol / vol_ma10.where(vol_ma10 != 0, 1))
            
            # Calculate volatility regime for adaptive weights
            vol_regime = price_std10 / price_std60
            w1 = 0.4 * (1 - vol_regime)  # More weight on IPE in low vol
            w2 = 0.3 * (1 + vol_regime)  # More weight on VVR in high vol
            w3 = 1 - w1 - w2             # MQS gets remaining weight
            
            # Normalize components
            ipe_norm = (ipe - ipe.rolling(window=20).mean()) / ipe.rolling(window=20).std()
            vvr_norm = (vvr - vvr.rolling(window=20).mean()) / vvr.rolling(window=20).std()
            mqs_norm = (mqs - mqs.rolling(window=20).mean()) / mqs.rolling(window=20).std()
            
            # Calculate final IMVF score
            result = (w1 * ipe_norm + 
                     w2 * vvr_norm + 
                     w3 * mqs_norm)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
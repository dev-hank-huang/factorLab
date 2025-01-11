from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('ALFM')
class ALFMCalculator(FactorCalculator):
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
            # Extract individual series
            price = close[key].ffill()
            vol = volume[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            
            # Calculate Liquidity Impact Ratio (LIR)
            price_change = abs(price - price.shift(1))
            daily_range = (high_price - low_price)
            lir = (vol * price_change) / daily_range.where(daily_range != 0, 1)
            
            # Calculate Flow Momentum Score (FMS)
            lir_ema5 = lir.ewm(span=5, adjust=False).mean()
            lir_ema20 = lir.ewm(span=20, adjust=False).mean()
            price_direction = np.sign(price - price.shift(5))
            fms = (lir_ema5 / lir_ema20.where(lir_ema20 != 0, 1)) * price_direction
            
            # Calculate Adaptive Regime Indicator (ARI)
            lir_std10 = lir.rolling(window=10).std()
            lir_std30 = lir.rolling(window=30).std()
            ari = lir_std10 / lir_std30.where(lir_std30 != 0, 1)
            
            # Calculate volume ratio
            vol_ma10 = vol.rolling(window=10).mean()
            volume_ratio = vol / vol_ma10.where(vol_ma10 != 0, 1)
            
            # Calculate final ALFM score
            alfm = fms * (1 + ari) * volume_ratio
            
            # Normalize and handle extreme values
            alfm_std = alfm.rolling(window=20).std()
            alfm_normalized = alfm / alfm_std.where(alfm_std != 0, 1)
            alfm_final = np.clip(alfm_normalized, -3, 3)
            
            # Store results
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = alfm_final
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
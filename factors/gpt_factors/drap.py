from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('DRAP')
class DRAPCalculator(FactorCalculator):
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
            
            # Calculate Price Velocity Profile (PVP)
            returns = price.pct_change()
            first_velocity = returns
            second_velocity = first_velocity.diff()
            vol_strength = vol / vol.rolling(window=10).mean()
            pvp = second_velocity * vol_strength
            
            # Calculate Flow Resistance Index (FRI)
            price_pressure = (price - open_price) / (high_price - low_price).where(
                (high_price - low_price) != 0, 1
            )
            vol_pressure = vol / vol.rolling(window=20).mean()
            fri = price_pressure * np.log1p(vol_pressure)
            
            # Calculate Phase Transition Signal (PTS)
            vol_5 = returns.rolling(window=5).std()
            vol_20 = returns.rolling(window=20).std()
            vol_ratio = vol_5 / vol_20.where(vol_20 != 0, 1)
            vol_ma10 = vol.rolling(window=10).mean()
            volume_ratio = vol / vol_ma10.where(vol_ma10 != 0, 1)
            momentum = (price / price.rolling(window=10).mean() - 1)
            pts = vol_ratio * volume_ratio * momentum
            
            # Adaptive signal combination based on market state
            market_state = abs(returns.rolling(window=20).mean()) / returns.rolling(window=20).std()
            
            # Dynamic weights based on market state
            w1 = 0.4 * (1 - market_state)  # PVP weight
            w2 = 0.3 * market_state        # FRI weight
            w3 = 1 - w1 - w2               # PTS weight
            
            # Normalize components
            pvp_norm = (pvp - pvp.rolling(window=20).mean()) / pvp.rolling(window=20).std()
            fri_norm = (fri - fri.rolling(window=20).mean()) / fri.rolling(window=20).std()
            pts_norm = (pts - pts.rolling(window=20).mean()) / pts.rolling(window=20).std()
            
            # Calculate final DRAP score with dynamic weighting
            result = (w1 * pvp_norm + 
                     w2 * fri_norm + 
                     w3 * pts_norm)
            
            # Apply non-linear transformation to amplify strong signals
            result = np.sign(result) * np.abs(result) ** 0.5
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
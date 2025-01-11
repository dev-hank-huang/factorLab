from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('CPD')
class CPDCalculator(FactorCalculator):
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
            
            # Calculate Relative Strength Divergence (RSD)
            stock_ma20 = price.rolling(window=20).mean()
            market_ma20 = close.mean(axis=1).rolling(window=20).mean()
            
            stock_strength = price / stock_ma20.where(stock_ma20 != 0, 1)
            market_strength = close.mean(axis=1) / market_ma20.where(market_ma20 != 0, 1)
            rsd = stock_strength - market_strength
            
            # Calculate Volume Impact Score (VIS)
            vol_ratio = vol / vol.rolling(window=20).mean()
            price_impact = (price - open_price) / (high_price - low_price).where(
                (high_price - low_price) != 0, 1
            )
            vis = np.sign(price_impact) * np.log1p(vol_ratio)
            
            # Calculate Pressure Imbalance Index (PII)
            returns = price.pct_change()
            uptick_volume = vol.where(returns > 0, 0)
            downtick_volume = vol.where(returns < 0, 0)
            
            rolling_total_volume = vol.rolling(window=5).sum()
            buy_pressure = uptick_volume.rolling(window=5).sum() / rolling_total_volume.where(
                rolling_total_volume != 0, 1
            )
            sell_pressure = downtick_volume.rolling(window=5).sum() / rolling_total_volume.where(
                rolling_total_volume != 0, 1
            )
            
            pii = (buy_pressure - sell_pressure) * vol_ratio
            
            # Normalize components
            rsd_norm = (rsd - rsd.rolling(window=20).mean()) / rsd.rolling(window=20).std()
            vis_norm = (vis - vis.rolling(window=20).mean()) / vis.rolling(window=20).std()
            pii_norm = (pii - pii.rolling(window=20).mean()) / pii.rolling(window=20).std()
            
            # Calculate market regime indicator
            volatility = returns.rolling(window=20).std()
            vol_regime = volatility / volatility.rolling(window=60).mean()
            
            # Dynamic weight adjustment based on volatility regime
            w1 = 0.4 * (1 - vol_regime)  # More weight on RSD in low vol
            w2 = 0.3 * (1 + vol_regime)  # More weight on VIS in high vol
            w3 = 1 - w1 - w2             # Remaining weight to PII
            
            # Combine signals with dynamic weights
            result = (w1 * rsd_norm + 
                     w2 * vis_norm + 
                     w3 * pii_norm)
            
            # Apply non-linear transformation
            result = np.sign(result) * np.power(np.abs(result), 0.6)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
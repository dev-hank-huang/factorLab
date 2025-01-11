from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('APEF')
class APEFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required price and volume data
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
        
        dfs = {}
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate Price Efficiency Ratio
            price_changes = price.pct_change()
            abs_price_changes = price_changes.abs()
            
            # Calculate rolling volatilities
            price_vol = price_changes.rolling(window=20).std()
            volume_vol = vol.rolling(window=20).std()
            
            # Normalize price changes and volume
            norm_price_changes = abs_price_changes / price_vol
            norm_volume = vol / volume_vol
            
            # Calculate Price Efficiency Ratio
            per = norm_price_changes / norm_volume.replace(0, np.nan)
            
            # Calculate Information Flow Rate
            def rolling_correlation(x, y, window=20):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            # Calculate rolling correlation between returns and sqrt volume
            sqrt_volume = np.sqrt(vol)
            ifr = rolling_correlation(abs_price_changes, sqrt_volume, window=20)
            
            # Calculate Market Impact Coefficient
            mic = price_changes / (sqrt_volume * price_vol)
            
            # Combine components with adaptive weights
            vol_regime = price_vol.rolling(window=60).rank(pct=True)
            
            # Adjust weights based on volatility regime
            per_weight = 0.4 * (1 + vol_regime)
            ifr_weight = 0.3 * (1 - vol_regime)
            mic_weight = 1 - per_weight - ifr_weight
            
            # Calculate final factor value
            result = (per_weight * per.fillna(0) +
                     ifr_weight * ifr.fillna(0) +
                     mic_weight * mic.fillna(0))
            
            # Apply non-linear transformation to reduce outliers
            result = np.sign(result) * np.log1p(np.abs(result))
            
            # Standardize the result
            result = (result - result.rolling(window=252).mean()) / result.rolling(window=252).std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
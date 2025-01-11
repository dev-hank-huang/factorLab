from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('ARTF')
class ARTFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required data
        close = data.get("price:close")
        high = data.get("price:high")
        low = data.get("price:low")
        volume = data.get("price:volume")
        
        # Ensure datetime index
        for df in [close, high, low, volume]:
            df.index = pd.to_datetime(df.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
        
        dfs = {}
        for key in close.columns:
            # Extract individual series
            c = close[key].ffill()
            h = high[key].ffill()
            l = low[key].ffill()
            v = volume[key].ffill()
            
            # Calculate Range Evolution Signal
            current_range = h - l
            historical_range = current_range.rolling(window=20).mean()
            
            def calculate_days_in_range(prices, window=20):
                rolling_high = prices.rolling(window=window).max()
                rolling_low = prices.rolling(window=window).min()
                return ((prices >= rolling_low) & (prices <= rolling_high)).astype(int).rolling(window).sum()
            
            days_in_range = calculate_days_in_range(c)
            decay_factor = 0.05
            
            res = (current_range / historical_range) * np.exp(-decay_factor * days_in_range)
            
            # Calculate Participation Shift Indicator
            volume_ma = v.rolling(window=20).mean()
            volume_ratio = v / volume_ma
            
            range_change = current_range.pct_change()
            range_ma = range_change.rolling(window=20).mean()
            
            psi = volume_ratio * (1 + np.abs(range_change - range_ma))
            
            # Calculate Extreme Price Elasticity
            range_center = (h + l) / 2
            range_width = h - l
            relative_position = (c - range_center) / range_width.replace(0, np.nan)
            
            returns = c.pct_change()
            up_volume = v.where(returns > 0, 0)
            down_volume = v.where(returns < 0, 0)
            volume_skew = (up_volume - down_volume) / (up_volume + down_volume)
            
            epe = relative_position * (1 + volume_skew)
            
            # Dynamic weight calculation
            volatility_regime = returns.rolling(window=20).std() / returns.rolling(window=60).std()
            volume_regime = volume_ratio.rolling(window=20).mean()
            
            # Adjust weights based on market conditions
            res_weight = 0.4 * np.clip(1 / volatility_regime, 0.5, 2)
            psi_weight = 0.35 * np.clip(volume_regime, 0.5, 2)
            epe_weight = 1 - res_weight - psi_weight
            
            # Calculate final factor value
            result = (res_weight * res.fillna(0) +
                     psi_weight * psi.fillna(0) +
                     epe_weight * epe.fillna(0))
            
            # Apply tanh transformation for outlier control
            result = np.tanh(result)
            
            # Standardize with expanding window
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
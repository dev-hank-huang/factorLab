from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('PDEF')
class PDEFCalculator(FactorCalculator):
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
            
            # Calculate returns and volatility
            returns = c.pct_change()
            price_volatility = returns.rolling(window=20).std()
            
            # Modified Trading Intensity Efficiency calculation
            volume_ma = v.rolling(window=20).mean()
            intensity = np.sqrt(v / volume_ma)
            
            # Replace autocorr with robust persistence calculation
            def calculate_persistence(series, window=10):
                """Calculate persistence using rolling regression coefficient"""
                series_lag = series.shift(1)
                def roll_reg(x, y):
                    if len(x) < 3:  # Minimum required for meaningful regression
                        return 0
                    try:
                        coef = np.cov(x, y)[0,1] / np.var(x)
                        return np.clip(coef, -1, 1)
                    except:
                        return 0
                
                persistence = pd.Series(index=series.index, dtype=float)
                for i in range(window, len(series)):
                    x = series_lag.iloc[i-window:i]
                    y = series.iloc[i-window:i]
                    persistence.iloc[i] = roll_reg(x.values, y.values)
                
                return persistence.fillna(0)
            
            intensity_persistence = calculate_persistence(intensity)
            
            tie = (returns / price_volatility) * intensity * (1 + intensity_persistence)
            
            # Calculate Price Discovery Quality with robust VWAP
            def calculate_robust_vwap(prices, volumes, window=20):
                """Calculate VWAP with safeguards against zero volumes"""
                running_sum = (prices * volumes).rolling(window=window).sum()
                vol_sum = volumes.rolling(window=window).sum()
                return running_sum / vol_sum.replace(0, np.nan)
            
            vwap = calculate_robust_vwap(c, v)
            price_range = (h - l).replace(0, np.nan)
            
            # Modified volume concentration calculation
            volume_concentration = (v / v.rolling(window=20).sum().replace(0, np.nan)).fillna(0)
            
            pdq = ((c - vwap) / price_range) * (1 + volume_concentration)
            
            # Calculate Market Depth Response with safeguards
            return_per_volume = returns / v.replace(0, np.nan)
            depth_ratio = v / volume_ma
            
            # Apply clipping to control extreme values
            mdr = np.clip(return_per_volume, -5, 5) * (1 + np.clip(depth_ratio, 0, 5))
            
            # Dynamic weight calculation with bounds
            volatility_state = np.clip(price_volatility / price_volatility.rolling(window=60).mean(), 0.5, 2)
            volume_state = np.clip(depth_ratio.rolling(window=20).mean(), 0.5, 2)
            
            # Bounded weight calculation
            tie_weight = 0.4 * np.clip(1 / volatility_state, 0.5, 2)
            pdq_weight = 0.35 * np.clip(volume_state, 0.5, 2)
            mdr_weight = 1 - tie_weight - pdq_weight
            
            # Calculate final factor value with robust handling
            result = (tie_weight * tie.fillna(0) +
                     pdq_weight * pdq.fillna(0) +
                     mdr_weight * mdr.fillna(0))
            
            # Modified outlier control using asinh transformation
            result = np.arcsinh(result)
            
            # Robust standardization
            result = (result - result.expanding().median()) / result.expanding().std().replace(0, 1)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
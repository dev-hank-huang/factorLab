from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('TIRPF')
class TIRPFCalculator(FactorCalculator):
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
            daily_vol = returns.rolling(window=20).std()
            
            # Calculate Volume Distribution Skew
            up_volume = v.where(returns > 0, 0)
            down_volume = v.where(returns < 0, 0)
            volume_skew = (up_volume - down_volume) / (up_volume + down_volume)
            price_deviation = returns.abs() / daily_vol
            
            vds = volume_skew * (1 + price_deviation)
            
            # Calculate Price Impact Resilience
            def calculate_vwap(prices, volumes):
                return (prices * volumes).rolling(window=20).sum() / volumes.rolling(window=20).sum()
            
            vwap = calculate_vwap(c, v)
            normalized_volume = v / v.rolling(window=20).mean()
            
            pir = ((c - vwap) / (v * daily_vol)).replace([np.inf, -np.inf], np.nan)
            
            # Calculate Imbalance Persistence Score
            def rolling_autocorr(series, window=20):
                return series.rolling(window).apply(
                    lambda x: pd.Series(x).autocorr(lag=1) if len(x.dropna()) > 1 else np.nan
                )
            
            volume_imbalance = (up_volume - down_volume) / v
            imbalance_autocorr = rolling_autocorr(volume_imbalance)
            
            range_normalized = (h - l) / (h.rolling(window=20).max() - l.rolling(window=20).min())
            
            ips = imbalance_autocorr * (1 + range_normalized)
            
            # Dynamic weight calculation
            volatility_regime = daily_vol / daily_vol.rolling(window=252).mean()
            volume_regime = normalized_volume.rolling(window=20).mean()
            
            # Adjust weights based on market conditions
            vds_weight = 0.4 * np.clip(1 / volatility_regime, 0.5, 2)
            pir_weight = 0.35 * np.clip(volume_regime, 0.5, 2)
            ips_weight = 1 - vds_weight - pir_weight
            
            # Calculate final factor value
            result = (vds_weight * vds.fillna(0) +
                     pir_weight * (-pir).fillna(0) +  # Negative because lower PIR indicates better resilience
                     ips_weight * ips.fillna(0))
            
            # Apply sigmoid transformation for outlier control
            result = 2 / (1 + np.exp(-result)) - 1
            
            # Standardize with expanding window
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
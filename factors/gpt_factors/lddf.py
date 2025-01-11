from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('LDDF')
class LDDFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required data
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
            returns = price.pct_change()
            
            # Calculate Liquidity Distribution Skew
            up_volume = vol.where(returns > 0, 0)
            down_volume = vol.where(returns < 0, 0)
            total_volume = vol.rolling(window=20).sum()
            
            def calc_price_efficiency(returns, window=20):
                cumret = returns.rolling(window).sum()
                path_length = returns.abs().rolling(window).sum()
                return np.abs(cumret) / path_length
            
            price_efficiency = calc_price_efficiency(returns)
            
            lds = ((up_volume - down_volume) / total_volume) * (1 + price_efficiency)
            
            # Calculate Participation Breadth Indicator
            volume_ma = vol.rolling(window=20).mean()
            effective_volume = vol.where(vol > volume_ma * 0.5, 0)
            
            def calc_time_concentration(volume, window=20):
                return (volume.rolling(window).std() / volume.rolling(window).mean()) ** 2
            
            time_concentration = calc_time_concentration(vol)
            
            pbi = (effective_volume.rolling(window=20).sum() / total_volume) * (1 + time_concentration)
            
            # Calculate Liquidity Resilience Score
            def calc_volume_persistence(volume, window=20):
                return volume.rolling(window).apply(
                    lambda x: pd.Series(x).autocorr(lag=1)
                )
            
            volume_persistence = calc_volume_persistence(vol)
            
            price_impact = returns.abs() / np.log1p(vol)
            price_impact_ratio = price_impact / price_impact.rolling(window=20).mean()
            
            lrs = volume_persistence * (1 + 1/price_impact_ratio)
            
            # Dynamic weight calculation
            liquidity_state = vol.rolling(window=20).mean() / vol.rolling(window=60).mean()
            volatility_state = returns.rolling(window=20).std() / returns.rolling(window=60).std()
            
            # Adjust weights based on market conditions
            lds_weight = 0.4 * np.clip(1/volatility_state, 0.5, 2)
            pbi_weight = 0.35 * np.clip(liquidity_state, 0.5, 2)
            lrs_weight = 1 - lds_weight - pbi_weight
            
            # Calculate final factor value
            result = (lds_weight * lds.fillna(0) +
                     pbi_weight * pbi.fillna(0) +
                     lrs_weight * lrs.fillna(0))
            
            # Apply asinh transformation for outlier control
            result = np.arcsinh(result)
            
            # Standardize with expanding window
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
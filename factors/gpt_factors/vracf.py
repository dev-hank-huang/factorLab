from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('VRACF')
class VRACFCalculator(FactorCalculator):
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
        
        # Calculate market-wide returns and volatility
        market_returns = close.pct_change().mean(axis=1)
        market_vol = market_returns.rolling(window=20).std()
        
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate returns
            returns = price.pct_change()
            
            # Calculate Regime Transition Indicator
            short_vol = returns.rolling(window=10).std()
            long_vol = returns.rolling(window=60).std()
            vol_change = short_vol.pct_change()
            
            rti = (short_vol / long_vol) * (1 + vol_change.abs())
            
            # Calculate Cross-Sectional Volatility Signal
            def rolling_correlation(x, y, window=20):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            market_correlation = rolling_correlation(returns, market_returns)
            individual_vol = returns.rolling(window=20).std()
            
            cvs = (individual_vol / market_vol) * (1 + market_correlation)
            
            # Calculate Volatility-Volume Coherence
            normalized_volume = (vol - vol.rolling(window=20).mean()) / vol.rolling(window=20).std()
            volume_smooth_change = normalized_volume.rolling(window=5).mean().pct_change()
            
            vol_vol_corr = rolling_correlation(individual_vol, normalized_volume)
            vvc = vol_vol_corr * (1 + volume_smooth_change.abs())
            
            # Calculate regime state
            regime_indicator = (short_vol / long_vol).rolling(window=20).rank(pct=True)
            
            # Dynamic weight adjustment
            base_rti_weight = 0.4
            base_cvs_weight = 0.35
            base_vvc_weight = 0.25
            
            # Adjust weights based on regime state
            rti_weight = base_rti_weight * (1 + regime_indicator)
            cvs_weight = base_cvs_weight * (1 + market_correlation.abs())
            vvc_weight = 1 - rti_weight - cvs_weight
            
            # Calculate final factor value
            result = (rti_weight * rti.fillna(0) +
                     cvs_weight * cvs.fillna(0) +
                     vvc_weight * vvc.fillna(0))
            
            # Apply non-linear transformation
            result = np.sign(result) * np.log1p(np.abs(result))
            
            # Standardize with expanding window
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
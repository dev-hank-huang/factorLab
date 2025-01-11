from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('VSTF')
class VSTFCalculator(FactorCalculator):
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
            
            # Calculate Volatility Term Structure Signal
            def calc_realized_vol(returns, window):
                return returns.rolling(window=window).std() * np.sqrt(252)
            
            vol_5d = calc_realized_vol(returns, 5)
            vol_20d = calc_realized_vol(returns, 20)
            vol_60d = calc_realized_vol(returns, 60)
            
            # Calculate volatility curve metrics
            vol_slope = (vol_5d - vol_60d) / vol_60d
            vol_curve = vol_20d - (vol_5d + vol_60d) / 2
            curve_change = vol_curve.diff()
            
            vts = (vol_5d / vol_60d) * (1 + np.abs(curve_change) / vol_60d)
            
            # Calculate Regime Transition Momentum
            vol_ma = vol_20d.rolling(window=60).mean()
            regime_diff = vol_20d - vol_ma
            
            def calc_regime_persistence(series, window=20):
                return series.rolling(window).apply(
                    lambda x: pd.Series(x).autocorr(lag=1)
                )
            
            regime_persistence = calc_regime_persistence(regime_diff)
            rtm = regime_diff * (1 + regime_persistence)
            
            # Calculate Volatility-Return Coherence
            def rolling_correlation(x, y, window=20):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            abs_returns = returns.abs()
            vol_changes = vol_20d.diff()
            vol_ret_corr = rolling_correlation(vol_changes, abs_returns)
            
            volume_intensity = (vol / vol.rolling(window=20).mean()).rolling(window=5).mean()
            vrc = vol_ret_corr * (1 + volume_intensity)
            
            # Dynamic weight calculation
            volatility_state = vol_20d / vol_20d.rolling(window=252).mean()
            regime_strength = np.abs(regime_diff / vol_ma)
            
            # Adjust weights based on market conditions
            vts_weight = 0.4 * np.clip(1 / volatility_state, 0.5, 2)
            rtm_weight = 0.35 * np.clip(regime_strength, 0.5, 2)
            vrc_weight = 1 - vts_weight - rtm_weight
            
            # Calculate final factor value
            result = (vts_weight * vts.fillna(0) +
                     rtm_weight * rtm.fillna(0) +
                     vrc_weight * vrc.fillna(0))
            
            # Apply modified softmax transformation for outlier control
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
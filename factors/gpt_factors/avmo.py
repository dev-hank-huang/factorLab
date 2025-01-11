from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('AVMO')
class AVMOCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Standard data preprocessing
        close = data.get("price:close")
        volume = data.get("price:volume")
        high = data.get("price:high")
        low = data.get("price:low")
        
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            
        dfs = {}
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            high_price = high[key].ffill()
            low_price = low[key].ffill()
            
            # 1. Trend/Momentum Component (50% weight)
            # Adaptive momentum using multiple timeframes
            mom_5 = price.pct_change(5)  # Short-term momentum
            mom_10 = price.pct_change(10)  # Medium-term momentum
            mom_20 = price.pct_change(20)  # Long-term momentum
            
            # Dynamic momentum weighting based on trend consistency
            trend_consistency = (mom_5.rolling(10).std() / mom_20.rolling(20).std()).clip(0.5, 2)
            momentum_score = (
                0.4 * mom_5 +
                0.35 * mom_10 +
                0.25 * mom_20
            ) * (1/trend_consistency)  # Reduce weight when trend is inconsistent
            
            # 2. Volume Confirmation Component (30% weight)
            # Volume trend and breakthrough detection
            vol_ma5 = vol.rolling(window=5).mean()
            vol_ma20 = vol.rolling(window=20).mean()
            
            # Volume breakthrough score
            vol_ratio = vol / vol_ma20
            vol_trend = vol_ma5 / vol_ma20
            volume_score = (
                0.6 * (vol_ratio - 1) +  # Recent volume surge
                0.4 * (vol_trend - 1)    # Volume trend
            )
            
            # 3. Risk Control Component (20% weight)
            # Volatility-adjusted range analysis
            true_range = pd.Series(np.maximum(high_price - low_price,
                                            np.maximum(abs(high_price - price.shift(1)),
                                                     abs(low_price - price.shift(1)))))
            
            atr_20 = true_range.rolling(window=20).mean()
            price_vol = price.rolling(window=20).std()
            
            # Risk score combining volatility and price range
            risk_score = -(
                0.5 * (true_range / atr_20 - 1) +  # Normalized range
                0.5 * (price_vol / price.rolling(window=60).std() - 1)  # Volatility ratio
            )
            
            # Normalize components
            def normalize(series):
                return (series - series.rolling(20).mean()) / series.rolling(20).std()
            
            momentum_norm = normalize(momentum_score)
            volume_norm = normalize(volume_score)
            risk_norm = normalize(risk_score)
            
            # Final composite signal
            result = (
                0.50 * momentum_norm +  # Trend/Momentum
                0.30 * volume_norm +    # Volume Confirmation
                0.20 * risk_norm       # Risk Control
            )
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
            
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
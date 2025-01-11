from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('PCVM')
class PCVMCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Fetch required price data
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
            # Extract individual stock data
            price = close[key].ffill()
            vol = volume[key].ffill()
            
            # Calculate Relative Price Oscillator (RPO)
            ma10 = price.rolling(window=10).mean()
            ma20 = price.rolling(window=20).mean()
            rpo = (ma20 - ma10) / ma20.where(ma20 != 0, np.nan)
            
            # Calculate Volume Cycle Ratio (VCR)
            vol_ma5 = vol.rolling(window=5).mean()
            vol_ma20 = vol.rolling(window=20).mean()
            vcr = vol_ma5 / vol_ma20.where(vol_ma20 != 0, np.nan)
            
            # Calculate Dynamic Frequency Transform (DFT)
            price_std30 = price.rolling(window=30).std()
            price_mean30 = price.rolling(window=30).mean()
            dft = price_std30 / price_mean30.where(price_mean30 != 0, np.nan)
            
            # Calculate market regime indicators
            volatility = price.pct_change().rolling(window=20).std()
            vol_ratio = vol / vol.rolling(window=20).mean()
            
            # Normalize components
            rpo_norm = (rpo - rpo.rolling(window=20).mean()) / rpo.rolling(window=20).std()
            vcr_norm = (vcr - vcr.rolling(window=20).mean()) / vcr.rolling(window=20).std()
            dft_norm = (dft - dft.rolling(window=20).mean()) / dft.rolling(window=20).std()
            
            # Calculate adaptive weights based on market conditions
            volatility_regime = volatility / volatility.rolling(window=60).mean()
            volume_regime = vol_ratio.rolling(window=10).mean()
            
            # Base weights
            w1 = 0.4  # RPO weight
            w2 = 0.35 # VCR weight
            w3 = 0.25 # DFT weight
            
            # Adjust weights based on market conditions
            regime_multiplier = np.clip(volatility_regime * volume_regime, 0.8, 1.2)
            
            w1_adj = w1 * regime_multiplier
            w2_adj = w2 * (2 - regime_multiplier)
            w3_adj = 1 - (w1_adj + w2_adj)  # Ensure weights sum to 1
            
            # Generate cycle phase signals
            cycle_signal = ((rpo_norm > 0) & (vcr_norm > 0)).astype(float)
            
            # Combine components into final factor
            result = (w1_adj * rpo_norm + 
                     w2_adj * vcr_norm + 
                     w3_adj * dft_norm) * cycle_signal
            
            # Apply volatility-based filtering
            result = result * (1 - (dft_norm > 2).astype(float))
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
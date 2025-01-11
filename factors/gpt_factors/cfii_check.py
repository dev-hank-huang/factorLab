from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('CapitalFlowImbalanceIndex')
class CapitalFlowImbalanceIndexCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Step 1: Retrieve necessary data
        close = data.get("price:close")
        volume = data.get("price:volume")
        high = data.get("price:high")
        low = data.get("price:low")
        
        # Ensure datetime format and resample if needed
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)

        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            high = high.resample(resample).max()
            low = low.resample(resample).min()

        # Step 2: Calculate volume imbalance ratio (VIR)
        volume_ma = volume.rolling(window=21).mean()
        vir = (volume / volume_ma - 1).fillna(0)

        # Step 3: Calculate price movement pressure (PMP)
        atr = (high - low).rolling(window=14).mean()
        delta_price = close.diff().fillna(0)
        pmp = (delta_price / atr).replace([np.inf, -np.inf], 0).fillna(0)

        # Step 4: Compute the CFII
        cfii = vir * pmp

        # Step 5: Normalize cross-sectionally
        cfii_zscore = (cfii - cfii.mean(axis=1)) / cfii.std(axis=1)

        # Wrap results into CustomDataFrame
        return CustomDataFrame(cfii_zscore)

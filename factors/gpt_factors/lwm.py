from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
from dataframe import CustomDataFrame
import numpy as np

@FactorRegistry.register('LWM')
class LWMCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Step 1: Retrieve core data
        close = data.get("price:close")
        volume = data.get("price:volume")
        market_cap = data.get("price:market_capital")
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        market_cap.index = pd.to_datetime(market_cap.index)

        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            market_cap = market_cap.resample(resample).last()

        # Step 2: Calculate Momentum
        momentum_period = 20  # Lookback period for momentum
        momentum = close - close.shift(momentum_period)

        # Step 3: Calculate Liquidity (Volume / Market Cap)
        liquidity = volume / market_cap

        # Step 4: Normalize Momentum and Liquidity
        momentum_z = (momentum - momentum.mean(axis=1, skipna=True)) / momentum.std(axis=1, skipna=True)
        liquidity_z = (liquidity - liquidity.mean(axis=1, skipna=True)) / liquidity.std(axis=1, skipna=True)

        # Step 5: Compute Liquidity-Weighted Momentum
        dfs = {}
        for col in close.columns:
            if liquidity_z[col].isnull().all() or momentum_z[col].isnull().all():
                lwm = pd.Series(index=close.index, dtype=float)
            else:
                lwm = momentum_z[col] / liquidity_z[col]
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][col] = lwm

        # Step 6: Format result into CustomDataFrame
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])

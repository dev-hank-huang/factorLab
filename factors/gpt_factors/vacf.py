from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
from dataframe import CustomDataFrame
import numpy as np

@FactorRegistry.register('VACF')
class VACFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Step 1: Retrieve core data
        close = data.get("price:close")
        open_price = data.get("price:open")
        volume = data.get("price:volume")
        close.index = pd.to_datetime(close.index)
        open_price.index = pd.to_datetime(open_price.index)
        volume.index = pd.to_datetime(volume.index)

        if resample.upper() != "D":
            close = close.resample(resample).last()
            open_price = open_price.resample(resample).first()
            volume = volume.resample(resample).sum()

        # Step 2: Calculate Net Capital Flow
        net_capital_flow = volume * (close - open_price)

        # Step 3: Calculate short-term rolling volatility
        returns = close.pct_change()
        rolling_volatility = returns.rolling(window=5).std()

        # Step 4: Compute Volatility-Adjusted Capital Flow
        dfs = {}
        for col in close.columns:
            if rolling_volatility[col].isnull().all() or net_capital_flow[col].isnull().all():
                vacf = pd.Series(index=close.index, dtype=float)
            else:
                vacf = net_capital_flow[col] / rolling_volatility[col]
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][col] = vacf

        # Step 5: Format result into CustomDataFrame
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])

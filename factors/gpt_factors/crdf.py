from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('CRDF')
class CRDFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required data and ensure proper datetime handling
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        # Convert index to datetime if not already
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        # Handle resampling with proper datetime alignment
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
        
        # Calculate returns with proper alignment
        returns = close.pct_change().fillna(0)
        
        def calculate_cross_correlation(data, window=20):
            """
            Calculate pairwise correlations with robust handling of timestamps
            """
            result = pd.DataFrame(index=data.index, columns=data.columns)
            
            # Use rolling windows with proper datetime alignment
            for i in range(window, len(data)):
                subset = data.iloc[i-window:i]
                # Calculate correlation matrix
                corr_matrix = subset.corr()
                # Extract upper triangle excluding diagonal
                upper_triangle = np.triu(corr_matrix.values, k=1)
                # Calculate mean correlation
                n_assets = len(data.columns)
                if n_assets > 1:
                    avg_corr = np.sum(upper_triangle) / (n_assets * (n_assets - 1) / 2)
                else:
                    avg_corr = 0
                result.iloc[i] = avg_corr
            
            return result.iloc[:, 0]  # Return series of average correlations
        
        dfs = {}
        for key in close.columns:
            price = close[key]
            vol = volume[key]
            
            # Ensure all timestamps are datetime objects
            price.index = pd.to_datetime(price.index)
            vol.index = pd.to_datetime(vol.index)
            
            # Calculate correlation dynamics
            rolling_corr = calculate_cross_correlation(returns, window=20)
            
            # Calculate median correlation with proper alignment
            rolling_median = rolling_corr.rolling(
                window=60,
                min_periods=30
            ).median()
            
            # Calculate correlation deviation
            corr_deviation = rolling_corr - rolling_median
            
            # Calculate volume-weighted impact
            normalized_volume = (vol / vol.rolling(window=20).mean()).fillna(1)
            
            # Calculate return impact
            return_series = price.pct_change()
            volatility = return_series.rolling(window=20).std()
            normalized_returns = return_series / volatility.replace(0, np.nan)
            
            # Combine components with proper alignment
            result = (0.4 * corr_deviation * normalized_volume +
                     0.3 * normalized_returns +
                     0.3 * rolling_corr)
            
            # Apply non-linear transformation for outlier control
            result = np.tanh(result)
            
            # Ensure proper standardization
            result = (result - result.expanding().mean()) / result.expanding().std().replace(0, 1)
            
            if 0 not in dfs:
                dfs[0] = {}
            dfs[0][key] = result
        
        # Ensure final dataframe has proper datetime index
        newdic = {0: pd.DataFrame(dfs[0], index=pd.to_datetime(close.index))}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
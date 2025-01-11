from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('DCPF')
class DCPFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # Get required data
        close = data.get("price:close")
        volume = data.get("price:volume")
        market_cap = data.get("price:market_capital")
        
        # Ensure datetime index
        for df in [close, volume, market_cap]:
            df.index = pd.to_datetime(df.index)
        
        # Handle resampling if needed
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            market_cap = market_cap.resample(resample).last()
        
        # Calculate market returns
        market_returns = (close * market_cap).sum(axis=1) / market_cap.sum(axis=1)
        market_returns = market_returns.pct_change()
        
        dfs = {}
        for key in close.columns:
            # Extract individual series
            price = close[key].ffill()
            returns = price.pct_change()
            
            # Calculate Co-Movement Evolution Score
            def rolling_correlation_diff(returns, market_returns, short_window=20, long_window=60):
                short_corr = returns.rolling(window=short_window).corr(market_returns)
                long_corr = returns.rolling(window=long_window).corr(market_returns)
                corr_std = short_corr.rolling(window=long_window).std()
                return (short_corr - long_corr) * (1 + np.abs(short_corr.diff() / corr_std))
            
            ces = rolling_correlation_diff(returns, market_returns)
            
            # Calculate Systematic Pressure Indicator
            def estimate_factor_exposure(returns, market_returns, window=60):
                return pd.Series(returns).rolling(window).cov(market_returns) / \
                       market_returns.rolling(window).var()
            
            beta = estimate_factor_exposure(returns, market_returns)
            excess_return = returns - market_returns
            factor_exposure = beta.abs() / beta.rolling(window=60).mean()
            
            spi = excess_return * (1 + factor_exposure)
            
            # Calculate Dynamic Response Coefficient
            def calculate_idiosyncratic_vol(returns, market_returns, beta, window=20):
                predicted_returns = beta * market_returns
                residuals = returns - predicted_returns
                return residuals.rolling(window=window).std()
            
            idio_vol = calculate_idiosyncratic_vol(returns, market_returns, beta)
            systematic_vol = (beta * market_returns).rolling(window=20).std()
            
            drc = beta * (1 + idio_vol / systematic_vol.replace(0, np.nan))
            
            # Dynamic weight calculation
            correlation_regime = ces.abs().rolling(window=60).mean()
            volatility_regime = returns.rolling(window=20).std() / returns.rolling(window=60).std()
            
            # Adjust weights based on market conditions
            ces_weight = 0.4 * np.clip(correlation_regime, 0.5, 2)
            spi_weight = 0.35 * np.clip(1 / volatility_regime, 0.5, 2)
            drc_weight = 1 - ces_weight - spi_weight
            
            # Calculate final factor value
            result = (ces_weight * ces.fillna(0) +
                     spi_weight * spi.fillna(0) +
                     drc_weight * drc.fillna(0))
            
            # Apply logistic transformation for outlier control
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
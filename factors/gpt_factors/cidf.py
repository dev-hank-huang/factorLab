from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('CIDF')
class CIDFCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # 獲取資料並確保日期格式正確
        close = data.get("price:close")
        volume = data.get("price:volume")
        market_cap = data.get("price:market_capital")
        
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        market_cap.index = pd.to_datetime(market_cap.index)
        
        # 處理重採樣
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
            market_cap = market_cap.resample(resample).last()
        
        # 預先計算常用的市場數據以提高效能
        returns = close.pct_change()
        volume_rank = volume.rank(axis=1, pct=True)
        liquid_returns = returns.where(volume_rank > 0.8)
        liquid_mean_return = liquid_returns.mean(axis=1)
        cross_sectional_mean = close.mean(axis=1)
        cross_sectional_std = close.std(axis=1)
        time_factor = np.sqrt(np.arange(1, len(close) + 1))
        
        dfs = {}
        dfs[0] = {}
        
        for key in close.columns:
            price = close[key].ffill()
            vol = volume[key].ffill()
            mcap = market_cap[key].ffill()
            
            # Lead-Lag Information Flow 計算
            stock_returns = returns[key]
            lif = pd.Series(stock_returns).rolling(20).corr(
                pd.Series(liquid_mean_return).shift(1)
            )
            volume_ratio = vol / vol.rolling(window=20).mean()
            lif_score = lif * (1 + volume_ratio)
            
            # Diffusion Speed Metric 計算
            price_deviation = (price - cross_sectional_mean) / cross_sectional_std
            dsm = price_deviation / time_factor
            
            # Information Shock Persistence 計算
            abs_returns = stock_returns.abs()
            return_autocorr = abs_returns.rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=1)
            )
            
            relative_range = (price.rolling(window=5).max() - price.rolling(window=5).min()) / \
                           (price.rolling(window=20).max() - price.rolling(window=20).min())
            
            isp = return_autocorr * (1 + relative_range)
            
            # 動態權重計算
            market_impact = (mcap / mcap.rolling(window=252).mean()).clip(0.5, 2)
            volume_impact = (volume_ratio.rolling(window=20).mean()).clip(0.5, 2)
            
            lif_weight = 0.4 * market_impact
            dsm_weight = 0.35 * volume_impact
            isp_weight = 1 - lif_weight - dsm_weight
            
            # 最終因子值計算
            result = (lif_weight * lif_score.fillna(0) +
                     dsm_weight * dsm.fillna(0) +
                     isp_weight * isp.fillna(0))
            
            # 非線性轉換和標準化
            result = np.sign(result) * np.log1p(np.abs(result))
            result = (result - result.expanding().mean()) / result.expanding().std()
            
            dfs[0][key] = result
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('RME')
class RMECalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # 獲取並預處理數據
        close = data.get("price:close")
        volume = data.get("price:volume")
        
        # 確保日期索引格式正確
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            volume = volume.resample(resample).sum()
        
        # 預先計算市場整體指標
        returns = close.pct_change()
        market_returns = close.mean(axis=1).pct_change(20)
        
        def calculate_entropy_vectorized(returns_series, window=10):
            """向量化的熵計算函數"""
            def single_entropy(x):
                hist, _ = np.histogram(x, bins='auto', density=True)
                prob_dist = hist / (hist.sum() + 1e-10)
                return -np.sum(prob_dist * np.log1p(prob_dist + 1e-10))
            
            return returns_series.rolling(window).apply(
                single_entropy, raw=True
            )
        
        def calculate_rank_corr_vectorized(returns):
            """向量化的排名相關性計算"""
            ranks = returns.rolling(5).mean()
            return ranks.rolling(5).corr(ranks.shift(1))
        
        dfs = {0: {}}
        
        # 並行處理所有股票
        for key in close.columns:
            try:
                price = close[key].ffill()
                vol = volume[key].ffill()
                
                # 計算 SRM
                stock_returns = price.pct_change(20)
                srm = stock_returns - market_returns
                
                # 計算 ETS
                stock_returns_short = price.pct_change()
                rolling_entropy = calculate_entropy_vectorized(stock_returns_short)
                entropy_change = rolling_entropy.diff()
                
                vol_intensity = vol / vol.rolling(window=20).mean()
                ets = entropy_change * vol_intensity
                
                # 計算 RVI
                rvi = calculate_rank_corr_vectorized(stock_returns_short)
                
                # 標準化處理
                def safe_standardize(x, window=20):
                    mean = x.rolling(window).mean()
                    std = x.rolling(window).std()
                    return ((x - mean) / std.where(std != 0, 1)).fillna(0)
                
                srm_norm = safe_standardize(srm)
                ets_norm = safe_standardize(ets)
                rvi_norm = safe_standardize(rvi)
                
                # 計算市場機制指標
                volatility = stock_returns_short.rolling(window=20).std()
                vol_regime = volatility / volatility.rolling(window=60).mean()
                
                # 動態權重調整
                w1 = 0.4 * (1 - vol_regime)
                w2 = 0.3 * (1 + vol_regime)
                w3 = 1 - w1 - w2
                
                # 組合信號
                result = (w1 * srm_norm + 
                         w2 * ets_norm + 
                         w3 * rvi_norm)
                
                # 使用安全的 sigmoid 轉換
                def safe_sigmoid(x):
                    clipped = np.clip(x, -100, 100)
                    return 2 / (1 + np.exp(-clipped)) - 1
                
                result = safe_sigmoid(result)
                dfs[0][key] = result.fillna(0)
                
            except Exception as e:
                dfs[0][key] = pd.Series(0, index=close.index)
                continue
        
        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
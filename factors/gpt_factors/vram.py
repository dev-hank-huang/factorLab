from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('VRAM')
class VRAMCalculator(FactorCalculator):
    def calculate(self, data, adjust_price=False, resample="D") -> CustomDataFrame:
        # 獲取價格數據
        close = data.get("price:close")
        high = data.get("price:high")
        low = data.get("price:low")
        volume = data.get("price:volume")
        
        # 確保日期索引格式正確
        close.index = pd.to_datetime(close.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        volume.index = pd.to_datetime(volume.index)
        
        if resample.upper() != "D":
            close = close.resample(resample).last()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
            
        # 預先計算市場層面的指標
        returns_all = close.pct_change().fillna(0)
        volatility_all = returns_all.rolling(window=20).std().fillna(0)
        hist_volatility_all = volatility_all.rolling(window=60).mean().fillna(volatility_all)
        
        dfs = {0: {}}
        
        def safe_standardize(x, window=20):
            """安全的標準化函數"""
            mean = x.rolling(window=window).mean()
            std = x.rolling(window=window).std()
            return ((x - mean) / std.where(std != 0, 1)).fillna(0)
        
        for key in close.columns:
            try:
                price = close[key].ffill()
                vol = volume[key].ffill()
                volatility = volatility_all[key]
                
                # 計算自適應收益率
                window = int(20 * (1 + np.log1p(volatility.mean())))
                window = min(max(window, 10), 40)
                ath = price.pct_change(window).fillna(0)
                
                # 計算波動率加權信號
                returns = returns_all[key]
                vol_adjusted_return = returns / volatility.where(volatility != 0, 1)
                var_95 = returns.rolling(window=60).quantile(0.05)
                var_ratio = abs(returns / var_95.where(var_95 != 0, 1))
                
                # 計算和標準化組件
                vws = (vol_adjusted_return * var_ratio).ewm(alpha=0.06).mean()
                vol_persistence = volatility.rolling(20).std() / volatility.rolling(20).mean()
                
                # 標準化所有組件
                ath_norm = safe_standardize(ath)
                vws_norm = safe_standardize(vws)
                rsi_norm = safe_standardize(vol_persistence)
                
                # 使用固定權重組合信號
                result = (0.4 * ath_norm + 
                         0.3 * vws_norm + 
                         0.3 * rsi_norm)
                
                # 應用非線性轉換
                result = np.sign(result) * np.log1p(np.abs(result))
                dfs[0][key] = result.fillna(0)
                
            except Exception as e:
                dfs[0][key] = pd.Series(0, index=close.index)
                continue

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
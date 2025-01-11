from ..registry import FactorCalculator, FactorRegistry
import pandas as pd
import numpy as np
from dataframe import CustomDataFrame

@FactorRegistry.register('NIFD')
class NIFDCalculator(FactorCalculator):
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
        
        # 處理重採樣
        if resample.upper() != "D":
            close = close.resample(resample).last()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()
            
        dfs = {}
        dfs[0] = {}

        def calculate_entropy(series, window=10):
            """修正後的熵計算函數"""
            try:
                # 使用 rolling 方法並確保數值有效
                normalized = series.rolling(window=window).apply(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
                ).fillna(0)
                
                # 確保所有值都在有效範圍內
                normalized = normalized.clip(1e-10, 1)
                
                # 計算熵值
                entropy = normalized.rolling(window=window).apply(
                    lambda x: -np.sum(np.where(x > 0, x * np.log(x), 0))
                ).fillna(0)
                
                return entropy
            except Exception as e:
                # 如果計算失敗，返回零序列
                return pd.Series(0, index=series.index)
            
        def safe_sigmoid(x):
            """
            實現一個安全的 sigmoid 轉換函數，避免數值溢出問題
            
            Args:
                x: 輸入數據
            Returns:
                轉換後的數據，範圍在 [-1, 1] 之間
            """
            # 首先將輸入值限制在合理範圍內
            clipped = np.clip(x, -100, 100)
            
            # 對於正值和負值分別處理，避免溢出
            pos_mask = clipped >= 0
            neg_mask = clipped < 0
            
            result = np.zeros_like(clipped, dtype=float)
            
            # 處理正值
            result[pos_mask] = 1 / (1 + np.exp(-clipped[pos_mask]))
            
            # 處理負值
            exp_x = np.exp(clipped[neg_mask])
            result[neg_mask] = exp_x / (1 + exp_x)
            
            # 轉換到 [-1, 1] 範圍
            return 2 * result - 1
        
        for key in close.columns:
            try:
                price = close[key].ffill()
                vol = volume[key].ffill()
                high_price = high[key].ffill()
                low_price = low[key].ffill()
                
                # 計算信息流率 (IFR)
                price_entropy = calculate_entropy(price)
                volume_entropy = calculate_entropy(vol)
                
                # 計算熵的變化
                price_entropy_change = price_entropy.diff().fillna(0)
                volume_entropy_change = volume_entropy.diff().fillna(0)
                
                ifr = price_entropy_change * volume_entropy_change
                
                # 計算網絡滲透分數 (NPS)
                vol_ma = vol.rolling(window=20).mean()
                trade_intensity = (vol / vol_ma).fillna(0)
                price_range = (high_price - low_price).fillna(0)
                price_dispersion = (price_range.rolling(window=10).std() / 
                                  price_range.rolling(window=20).mean()).fillna(0)
                price_momentum = price.pct_change(10).fillna(0)
                
                nps = trade_intensity * price_dispersion * np.sign(price_momentum)
                
                # 計算自適應響應函數 (ARF)
                returns = price.pct_change().fillna(0)
                vol_current = returns.rolling(window=5).std().fillna(0)
                vol_historical = returns.rolling(window=60).std().fillna(0)
                market_state = (vol_current / vol_historical.where(vol_historical != 0, 1)).fillna(0)
                
                info_rate = np.log1p((vol / vol_ma.where(vol_ma != 0, 1))).fillna(0)
                
                arf = np.tanh(market_state * info_rate)
                
                # 標準化組件
                def safe_standardize(x, window=20):
                    mean = x.rolling(window=window).mean()
                    std = x.rolling(window=window).std()
                    return ((x - mean) / std.where(std != 0, 1)).fillna(0)
                
                ifr_norm = safe_standardize(ifr)
                nps_norm = safe_standardize(nps)
                arf_norm = safe_standardize(arf)
                
                # 計算自適應權重
                info_clarity = (abs(ifr_norm) / (abs(nps_norm) + abs(arf_norm) + 1e-10)).fillna(0)
                w1 = 0.4 * (1 + info_clarity)
                w2 = 0.3 * (1 - info_clarity)
                w3 = 1 - w1 - w2
                
                # 組合組件並進行非線性轉換
                result = (w1 * ifr_norm + w2 * nps_norm + w3 * arf_norm)
                result = safe_sigmoid(result).fillna(0)
                
                dfs[0][key] = result
                
            except Exception as e:
                # 如果處理某支股票時發生錯誤，將其結果設為零序列
                dfs[0][key] = pd.Series(0, index=close.index)
                continue

        newdic = {0: pd.DataFrame(dfs[0], index=close.index)}
        ret = [newdic[0]]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]
        
        return CustomDataFrame(ret[0])
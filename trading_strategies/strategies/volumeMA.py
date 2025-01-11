from typing import Dict, Any
from ..base import TradingStrategy
from ..registry import StrategyRegistry
import pandas as pd
import numpy as np

@StrategyRegistry.register("VOLUME_MA_BREAKTHROUGH")
class ImprovedVolumeMAStrategy(TradingStrategy):
    """
    改進版成交量均線策略
    
    主要改進：
    1. 動態調整參數 - 基於市場波動性
    2. 多重時間框架分析
    3. 趨勢強度確認
    4. 智能止盈止損
    5. 靈活的持倉管理
    """
    
    def __init__(self):
        super().__init__(
            name="improved_volume_ma",
            description="改進版成交量均線策略，整合多重確認機制"
        )
    
    def calculate_market_volatility(self, close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """計算市場波動性"""
        returns = close.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_trend_strength(self, close: pd.DataFrame, ma_period: int) -> pd.DataFrame:
        """計算趨勢強度指標"""
        ma = close.rolling(window=ma_period).mean()
        short_ma = close.rolling(window=ma_period // 2).mean()
        trend_strength = (short_ma - ma) / ma * 100
        return trend_strength.fillna(0)
    
    def calculate_volume_profile(self, volume: pd.DataFrame, base_period: int) -> tuple:
        """計算成交量特徵"""
        # 計算成交量移動平均
        volume_ma = volume.rolling(window=base_period).mean()
        
        # 計算成交量波動率
        volume_std = volume.rolling(window=base_period).std()
        
        # 計算相對成交量強度
        relative_volume = volume / volume_ma
        
        # 計算成交量趨勢
        volume_trend = volume.rolling(window=base_period).mean() / \
                      volume.rolling(window=base_period*2).mean()
                      
        return volume_ma, volume_std, relative_volume, volume_trend
    
    def calculate_dynamic_thresholds(self, volatility: pd.DataFrame) -> tuple:
        """基於波動率計算動態閾值"""
        base_deviation = 10  # 基礎偏離率
        base_volume_mult = 1.5  # 基礎成交量倍數
        
        # 根據波動率調整閾值
        deviation_threshold = base_deviation * (1 + volatility)
        volume_threshold = base_volume_mult * (1 + volatility * 0.5)
        
        return deviation_threshold, volume_threshold
    
    def generate_signals(self, data: Any, **params) -> pd.DataFrame:
        """生成交易信號"""
        # 獲取價格和成交量數據
        close = data.get('price:close')
        volume = data.get('price:volume')
        
        # 確保索引格式一致
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        # 基礎參數處理
        ma_period = int(params.get('ma_period', 20))
        min_holding = int(params.get('min_holding', 5))
        max_holding = int(params.get('max_holding', 20))
        
        # 計算市場波動性
        volatility = self.calculate_market_volatility(close)
        
        # 計算價格指標
        price_ma = close.rolling(window=ma_period).mean()
        trend_strength = self.calculate_trend_strength(close, ma_period)
        
        # 計算成交量指標
        volume_ma, volume_std, relative_volume, volume_trend = \
            self.calculate_volume_profile(volume, ma_period)
        
        # 計算動態閾值
        deviation_threshold, volume_threshold = \
            self.calculate_dynamic_thresholds(volatility)
        
        # 進場條件
        price_condition = (
            (close < price_ma) &  # 價格在均線下方
            (close > close.rolling(window=5).min()) &  # 價格開始反彈
            (trend_strength > -15)  # 趨勢不過於弱勢
        )
        
        volume_condition = (
            (relative_volume > volume_threshold) &  # 成交量突破
            (volume_trend > 1)  # 成交量趨勢向上
        )
        
        # 最終進場信號
        entries = price_condition & volume_condition
        
        # 出場條件
        profit_condition = close > (price_ma * (1 + deviation_threshold/100))
        loss_condition = close < (price_ma * (1 - deviation_threshold/100))
        trend_reversal = trend_strength < -10
        
        exits = profit_condition | loss_condition | trend_reversal
        
        # 生成持倉訊號
        if hasattr(entries, 'hold_until'):
            signals = entries.hold_until(exits)
        else:
            signals = pd.DataFrame(False, index=close.index, columns=close.columns)
            holding_periods = pd.DataFrame(0, index=close.index, columns=close.columns)
            
            for col in signals.columns:
                position = False
                holding_days = 0
                
                for i in range(len(signals)):
                    if entries.iloc[i][col]:
                        position = True
                        holding_days = 0
                    elif exits.iloc[i][col] or holding_days >= max_holding:
                        if holding_days >= min_holding:  # 確保最小持倉時間
                            position = False
                            holding_days = 0
                    
                    if position:
                        holding_days += 1
                        
                    signals.iloc[i][col] = position
                    holding_periods.iloc[i][col] = holding_days
        
        return signals
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """返回優化後的策略參數"""
        return {
            'ma_period': 20,           # 均線週期
            'min_holding': 5,          # 最小持倉天數
            'max_holding': 20,         # 最大持倉天數
            'base_deviation': 10,      # 基礎偏離率
            'volume_multiplier': 1.5   # 成交量倍數
        }
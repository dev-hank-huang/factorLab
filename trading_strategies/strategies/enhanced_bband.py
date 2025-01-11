from typing import Dict, Any
from ..base import TradingStrategy
from ..registry import StrategyRegistry
import pandas as pd
import numpy as np

@StrategyRegistry.register("ENHANCED_BBAND")
class EnhancedBollingerBandsStrategy(TradingStrategy):
    def __init__(self):
        super().__init__(
            name="enhanced_bband",
            description="增強版布林通道策略，整合多重指標確認"
        )
    
    def calculate_trend_strength(self, close: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
        """計算趨勢強度"""
        ma = close.rolling(window=ma_period).mean()
        trend_strength = (close - ma) / ma * 100
        return trend_strength.fillna(0)  # 填充NaN值
    
    def calculate_dynamic_bands(self, close: pd.DataFrame, volume: pd.DataFrame,
                              base_period: int = 20) -> tuple:
        """計算動態布林通道"""
        # 使用更穩定的波動率計算
        volatility = close.rolling(window=base_period).std() / close.rolling(window=base_period).mean()
        volume_ratio = volume / volume.rolling(window=base_period).mean()
        
        # 限制動態調整範圍
        dynamic_dev = (2 + (volatility * volume_ratio).clip(0, 0.5)).fillna(2)
        
        ma = close.rolling(window=base_period).mean()
        std = close.rolling(window=base_period).std()
        
        upper = ma + (std * dynamic_dev)
        lower = ma - (std * dynamic_dev)
        
        return upper.fillna(method='ffill'), ma.fillna(method='ffill'), lower.fillna(method='ffill')

    def generate_signals(self, data: Any, **params) -> pd.DataFrame:
        """生成交易信號"""
        # 確保所有輸入數據的索引格式一致
        close = data.get('price:close')
        volume = data.get('price:volume')
        
        close.index = pd.to_datetime(close.index)
        volume.index = pd.to_datetime(volume.index)
        
        # 參數處理
        base_period = int(params.get('base_period', 20))
        trend_threshold = float(params.get('trend_threshold', 5))
        profit_target = float(params.get('profit_target', 0.1))
        stop_loss = float(params.get('stop_loss', 0.05))
        
        # 計算技術指標
        upper, middle, lower = self.calculate_dynamic_bands(close, volume, base_period)
        trend = self.calculate_trend_strength(close, base_period)
        rsi = data.indicator('RSI', timeperiod=14)
        
        # 成交量確認
        volume_ma = volume.rolling(window=base_period).mean()
        volume_ratio = (volume / volume_ma).fillna(0)
        volume_confirm = volume_ratio > 1.5
        
        # 進場條件
        entries = (
        (close > upper) &
        (close.shift(1) <= upper.shift(1)) &
        (volume > volume_ma * 1.2)  # 降低成交量要求
        ).fillna(False)
        
        # 出場條件
        exits = (
        (close < middle) |
        (close < close.shift(1) * (1 - stop_loss))  
        ).fillna(False)
        
        # 使用hold_until生成最終信號
        if hasattr(entries, 'hold_until'):
            return entries.hold_until(exits)
        else:
            # 手動實現hold_until邏輯
            positions = pd.DataFrame(False, index=close.index, columns=close.columns)
            for col in positions.columns:
                position = False
                for i in range(len(positions)):
                    if entries.iloc[i][col]:
                        position = True
                    elif exits.iloc[i][col]:
                        position = False
                    positions.iloc[i][col] = position
            return positions

    def get_optimal_parameters(self) -> Dict[str, Any]:
        """返回台股市場優化後的參數"""
        return {
            'base_period': 20,
            'trend_threshold': 5,
            'profit_target': 0.1,
            'stop_loss': 0.05
        }
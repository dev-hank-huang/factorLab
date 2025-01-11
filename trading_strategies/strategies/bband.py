from ..base import TradingStrategy
from ..registry import StrategyRegistry
from typing import Dict, Any
import pandas as pd

@StrategyRegistry.register("BOLLINGER_BREAKTHROUGH")
class BollingerBreakthroughStrategy(TradingStrategy):
    """布林通道突破策略"""
    
    def __init__(self):
        super().__init__(
            name="bollinger_breakthrough",
            description="布林通道突破交易策略"
        )
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """驗證策略參數"""
        required_params = {'timeperiod', 'nbdevup', 'nbdevdn'}
        return all(param in params for param in required_params)
    
    def generate_signals(self, data: Any, **params) -> pd.DataFrame:
        """生成布林通道突破策略信號"""
        try:
            close = data.get('price:close')
            volume = data.get('price:volume')
            
            # 計算布林通道
            bands = data.indicator(
                'BBANDS',
                timeperiod=params.get('timeperiod', 20),
                nbdevup=params.get('nbdevup', 2.0),
                nbdevdn=params.get('nbdevdn', 2.0)
            )
            
            # 計算成交量確認
            volume_window = params.get('volume_window', 20)
            vol_ma = volume.rolling(window=volume_window).mean()
            volume_confirm = volume > vol_ma
            
            # 生成進場信號
            entries = (close > bands[0]) & (close.shift(1) <= bands[0]) & volume_confirm
            
            # 生成出場信號
            exits = (close < bands[2]) | (close < bands[1])
            
            # 計算持有信號
            signals = entries.hold_until(exits)
            
            # 確保返回DataFrame格式
            signals = pd.DataFrame(signals, index=close.index, columns=close.columns)
            
            return signals
            
        except Exception as e:
            print(f"布林通道策略生成信號時發生錯誤: {str(e)}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            return pd.DataFrame()

@StrategyRegistry.register("BOLLINGER_REVERSAL")
class BollingerReversalStrategy(TradingStrategy):
    """布林通道反轉策略"""
    
    def __init__(self):
        super().__init__(
            name="bollinger_reversal",
            description="布林通道反轉交易策略"
        )
    
    def generate_signals(self, data: Any, **params) -> pd.DataFrame:
        """生成交易信號"""
        close = data.get('price:close')
        volume = data.get('price:volume')
        
        # 計算布林通道
        bands = data.indicator(
            'BBANDS',
            timeperiod=params.get('timeperiod', 20),
            nbdevup=params.get('nbdevup', 2),
            nbdevdn=params.get('nbdevdn', 2)
        )
        
        # 生成反轉信號
        entries = (close < bands[2]) & (close.shift(1) >= bands[2])
        exits = (close > bands[0]) | (close > bands[1])
        
        return entries.hold_until(exits)
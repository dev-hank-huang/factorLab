from typing import List, Dict, Any
import pandas as pd
from ..registry import StrategyRegistry
from ..base import TradingStrategy

class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self._registry = StrategyRegistry
        
    def get_strategy(self, strategy_name: str) -> TradingStrategy:
        """獲取策略實例"""
        strategy_class = self._registry.get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Strategy {strategy_name} not found")
        return strategy_class()
    
    def generate_signals(self, 
                        strategy_name: str, 
                        data: Any, 
                        **params) -> pd.DataFrame:
        """生成單一策略信號"""
        strategy = self.get_strategy(strategy_name)
        return strategy.generate_signals(data, **params)
    
    def generate_combined_signals(self,
                                strategies: List[Dict[str, Any]],
                                data: Any,
                                combination_type: str = "AND") -> pd.DataFrame:
        """生成組合策略信號"""
        if not strategies:
            raise ValueError("No strategies specified")
        
        signals = []
        for strategy_config in strategies:
            strategy_name = strategy_config['name']
            params = strategy_config.get('params', {})
            signal = self.generate_signals(strategy_name, data, **params)
            signals.append(signal)
        
        combined = pd.concat(signals, axis=1)
        if combination_type == "AND":
            return combined.all(axis=1)
        elif combination_type == "OR":
            return combined.any(axis=1)
        else:
            raise ValueError(f"Invalid combination type: {combination_type}")
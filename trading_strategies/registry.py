from typing import Dict, Type
from .base import TradingStrategy

class StrategyRegistry:
    """策略註冊管理器"""
    
    _strategies: Dict[str, Type[TradingStrategy]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """策略註冊裝飾器"""
        def wrapper(strategy_class: Type[TradingStrategy]):
            strategy_name = name or strategy_class.__name__
            cls._strategies[strategy_name.upper()] = strategy_class
            return strategy_class
        return wrapper
    
    @classmethod
    def get_strategy(cls, name: str) -> Type[TradingStrategy]:
        """獲取已註冊的策略類"""
        return cls._strategies.get(name.upper())
    
    @classmethod
    def list_registered_strategies(cls):
        """列出所有已註冊的策略"""
        return sorted(list(cls._strategies.keys()))
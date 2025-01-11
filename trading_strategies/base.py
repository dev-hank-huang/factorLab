from abc import ABC, abstractmethod
from typing import Dict, Any
from dataframe import CustomDataFrame

class TradingStrategy(ABC):
    """交易策略的抽象基類"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        
    @abstractmethod
    def generate_signals(self, data: Any, **params) -> CustomDataFrame:
        """生成交易信號的抽象方法"""
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """驗證策略參數的預設方法"""
        return True
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
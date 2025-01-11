from abc import ABC, abstractmethod
from typing import Dict, Type
from dataframe import CustomDataFrame
import logging
from typing import Dict, Any
import pandas as pd

class FactorCalculator(ABC):
    """因子計算的抽象基類"""
    
    @abstractmethod
    def calculate(self, data_api, adjust_price=False, resample="D") -> CustomDataFrame:
        """計算因子值"""
        pass
    
    def calculate_with_shared_data(self, shared_data: Dict[str, Any]) -> CustomDataFrame:
        """使用共享數據計算因子"""
        class SharedDataAPI:
            def __init__(self, price_data: Dict[str, Any]):
                self._price_data = price_data
                self.logger = logging.getLogger('SharedDataAPI')
                
            def get(self, data_type: str) -> CustomDataFrame:
                try:
                    if ':' not in data_type:
                        self.logger.error(f"無效的數據類型格式: {data_type}")
                        return None
                        
                    category, item = data_type.split(':')
                    if category == 'price':
                        if item not in self._price_data:
                            self.logger.error(f"找不到價格數據項: {item}")
                            return None
                            
                        df = self._price_data[item]
                        # 確保數據是 DataFrame
                        if not isinstance(df, pd.DataFrame):
                            df = pd.DataFrame(df)
                            
                        # 確保索引是 datetime
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                            
                        return CustomDataFrame(df)
                        
                except Exception as e:
                    self.logger.error(f"獲取數據時發生錯誤: {str(e)}")
                    return None
                    
        try:
            # 創建數據訪問對象
            data_api = SharedDataAPI(shared_data)
            # 調用實際的計算方法
            return self.calculate(data_api)
        except Exception as e:
            logging.error(f"因子計算發生錯誤: {str(e)}")
            return None

class FactorRegistry:
    """因子註冊管理器"""
    
    _calculators: Dict[str, Type[FactorCalculator]] = {}
    
    @classmethod
    def register(cls, name: str):
        def wrapper(calculator: Type[FactorCalculator]):
            cls._calculators[name.upper()] = calculator
            return calculator
        return wrapper
    
    @classmethod
    def get_calculator(cls, name: str) -> Type[FactorCalculator]:
        return cls._calculators.get(name.upper())
    
    @classmethod
    def list_registered_factors(cls):
        """列出所有已註冊的因子"""
        return sorted(list(cls._calculators.keys()))
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataframe import CustomDataFrame
import logging
import pandas as pd

class FactorCalculator(ABC):
    @abstractmethod
    def calculate(self, data, adjust_price=False, resample="D"):
        pass
    
    def calculate_with_shared_data(self, shared_data: Dict[str, Any]) -> CustomDataFrame:
        """使用共享數據計算因子"""
        class SharedDataAPI:
            def __init__(self, price_data: Dict[str, Any]):
                self._price_data = price_data
                self.logger = logging.getLogger('SharedDataAPI')
                
            def get(self, data_type: str) -> CustomDataFrame:
                """模擬原始 data.get() 方法的行為"""
                try:
                    if ':' not in data_type:
                        self.logger.error(f"無效的數據類型格式: {data_type}")
                        return None
                        
                    category, item = data_type.split(':')
                    if category == 'price':
                        if item not in self._price_data:
                            self.logger.error(f"找不到價格數據項: {item}")
                            return None
                            
                        data = self._price_data[item]
                        if not isinstance(data, pd.DataFrame):
                            data = pd.DataFrame(data)
                            
                        # 確保索引是 datetime
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data.index = pd.to_datetime(data.index)
                            
                        return CustomDataFrame(data)
                        
                    self.logger.error(f"不支持的數據類別: {category}")
                    return None
                    
                except Exception as e:
                    self.logger.error(f"獲取數據時發生錯誤: {str(e)}")
                    return None
        
        try:
            # 建立模擬的數據訪問對象
            data_api = SharedDataAPI(shared_data)
            
            # 調用實際的計算方法
            return self.calculate(data_api)
        except Exception as e:
            logging.error(f"因子計算發生錯誤: {str(e)}")
            return None
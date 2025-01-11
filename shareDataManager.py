from multiprocessing import Manager
import numpy as np
import pandas as pd

class SharedDataManager:
    def __init__(self):
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
   
    def initialize_shared_data(self, data):
        """Initialize shared data using Manager dict instead of SharedMemory"""
        try:
            # 準備價格數據
            price_data = {
                'close': data.get('price:close'),
                'open': data.get('price:open'),
                'high': data.get('price:high'),
                'low': data.get('price:low'),
                'volume': data.get('price:volume'),
                'market_capital': data.get('price:market_capital')
            }
            
            # 將每個 DataFrame 轉換為可序列化的格式
            for key, df in price_data.items():
                if df is not None:
                    self.shared_dict[key] = {
                        'values': df.values.tolist(),  # numpy array 轉為 list
                        'index': df.index.astype(str).tolist(),  # 時間索引轉為字符串
                        'columns': df.columns.tolist()
                    }
            
            return dict(self.shared_dict)  # 返回普通字典的副本
            
        except Exception as e:
            print(f"初始化共享數據時發生錯誤: {str(e)}")
            return {}
   
    def get_shared_data(self, shared_data_info):
        """Reconstruct DataFrames from shared data"""
        try:
            result = {}
            for key, data_info in shared_data_info.items():
                if isinstance(data_info, dict):
                    # 重建 DataFrame
                    df = pd.DataFrame(
                        data=data_info['values'],
                        index=pd.to_datetime(data_info['index']),
                        columns=data_info['columns']
                    )
                    result[key] = df
                    
            return result
            
        except Exception as e:
            print(f"重建數據時發生錯誤: {str(e)}")
            return {}
   
    def cleanup(self):
        """Clean up shared resources"""
        if hasattr(self, 'manager'):
            self.manager.shutdown()
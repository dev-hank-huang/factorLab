from get_data import Data
from database import Database
from backtest import *
from operations import *
from datetime import datetime
import pandas as pd
from dataframe import CustomDataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict

from abc import ABC, abstractmethod

# 定義一個抽象基類 FactorAnalysis，它將為所有因子分析方法提供一個共同的接口
# 這個類別將處理初始化操作，並定義一個抽象方法 analyze 供子類實現
class FactorAnalysis(ABC):
    def __init__(self, factor_name_list, factor_asc_dict, quantile, all_factor_df_dict):
        self.factor_name_list = factor_name_list
        self.factor_asc_dict = factor_asc_dict
        self.quantile = quantile
        self.all_factor_df_dict = all_factor_df_dict

    @abstractmethod
    def analyze(self):
        """
        抽象方法，子類必須實現此方法以執行具體的因子分析。
        """
        pass

# 這個類別實現了對單一因子的分析，根據因子值的大小和指定的排序方向進行分割
class SingleFactorAnalysis(FactorAnalysis):
    def analyze(self):
        factor_df = self.all_factor_df_dict[self.factor_name_list[0]]
        factor_asc = self.factor_asc_dict[self.factor_name_list[0]]
        return factor_df.divide_slice(self.quantile, factor_asc)

# 具體類別TwoFactorInterpolatedAnalysis，這個類別將實現對兩個因子的加權內插分析
class TwoFactorInterpolatedAnalysis(FactorAnalysis):
    def __init__(self, factor_name_list, factor_asc_dict, quantile, all_factor_df_dict, method='ranked', factor_ratio_dict=None):
        super().__init__(factor_name_list, factor_asc_dict, quantile, all_factor_df_dict)
        self.method = method
        self.factor_ratio_dict = factor_ratio_dict or {name: 1.0/len(factor_name_list) for name in factor_name_list}
    
    def analyze(self):
        factor_1, factor_2 = self.factor_name_list
        # 檢查是否存在因子資料
        if factor_1 not in self.all_factor_df_dict or factor_2 not in self.all_factor_df_dict:
            raise ValueError("Factor data not found in the provided dictionary")
        
        # 擷取因子數據框
        factor_1_df = self.all_factor_df_dict[factor_1]
        factor_2_df = self.all_factor_df_dict[factor_2]

        # 計算兩個因子的加權內插或排名分數
        factor_1_scores = self.calculate_interpolated_scores(factor_1_df, self.factor_asc_dict[factor_1], self.method)
        factor_2_scores = self.calculate_interpolated_scores(factor_2_df, self.factor_asc_dict[factor_2], self.method)

        # 加權合成分數
        weighted_scores = factor_1_scores * self.factor_ratio_dict[factor_1] + factor_2_scores * self.factor_ratio_dict[factor_2]
        return weighted_scores.divide_slice(self.quantile)


# 具體類別 TwoFactorFilterAnalysis
# 這個類別實現了雙因子過濾篩選方法，強調第一個因子並減弱第二個因子的影響
class TwoFactorFilterAnalysis(FactorAnalysis):
    def analyze(self):
        factor_1, factor_2 = self.factor_name_list
        factor_1_df = self.all_factor_df_dict[factor_1]
        factor_2_df = self.all_factor_df_dict[factor_2]

        factor_1_asc = self.factor_asc_dict[factor_1]
        factor_2_asc = self.factor_asc_dict[factor_2]

        # 將第一個因子按照分位數進行切割
        factor_1_slice_dict = factor_1_df.divide_slice(self.quantile, factor_1_asc)

        # 應用過濾機制
        factor1_mask_factor2 = {}
        for q, df in factor_1_slice_dict.items():
            key = f"{q}_MASK_factor2"
            value = self.mask(df, factor_2_df)
            factor1_mask_factor2[key] = value

        result = {}
        for i in range(self.quantile):
            key = f"Quantile_{i+1}"
            mask_key = f"Quantile_{i+1}_MASK_factor2"
            sliced_data = factor1_mask_factor2[mask_key].divide_slice(self.quantile, factor_2_asc)
            result[key] = sliced_data["Quantile_" + str(i + 1)]

        return result
      
# 具體類別 TwoFactorANDAnalysis
# 這個類別將兩個因子的資料框按分位數分割後進行AND交集運算
class TwoFactorANDAnalysis(FactorAnalysis):
    def analyze(self):
        factor_1, factor_2 = self.factor_name_list
        factor_1_df = self.all_factor_df_dict[factor_1]
        factor_2_df = self.all_factor_df_dict[factor_2]

        factor_1_asc = self.factor_asc_dict[factor_1]
        factor_2_asc = self.factor_asc_dict[factor_2]

        factor_1_slices = factor_1_df.divide_slice(self.quantile, factor_1_asc)
        factor_2_slices = factor_2_df.divide_slice(self.quantile, factor_2_asc)

        result = {}
        for i in range(self.quantile):
            key = f"Quantile_{i+1}"
            value = factor_1_slices[f"Quantile_{i+1}"] & factor_2_slices[f"Quantile_{i+1}"]
            result[key] = value

        return result


# 呼叫範例
if __name__ == '__main__':
    # 假設 all_factor_df_dict 是已經被正確初始化的字典
    all_factor_df_dict = {
        'roe': CustomDataFrame(...),
        'pb': CustomDataFrame(...)
    }

    # 初始化具體的分析類別並提供必要的參數
    filter_analysis = TwoFactorFilterAnalysis(['roe', 'pb'], {'roe': True, 'pb': False}, 4, all_factor_df_item)
    and_analysis = TwoFactorANDAnalysis(['roe', 'pb'], {'roe': True, 'pb': False}, 4, all_factor_df_dict)

    # 執行分析
    filter_result = filter_analysis.analyze()
    and_result = and_analysis.analyze()

    # 輸出或進一步處理結果
    print(filter_result)
    print(and_result)

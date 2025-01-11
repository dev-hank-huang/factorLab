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



# 雙因子切割(加權內插/排名法用)
def cal_factor_sum_df_interpolated(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    method: str = 'ranked',
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
):
    """
    Args:
        factor_name_list (list): 包含多個因子的名稱，例如: factor_name_list = ['roe','pb']
        factor_ratio_dict (dict): 包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        factor_asc_dict (dict): 一個字典，包含多個因子的排序方向
        quantile (positive-int): 打算將因子切割成幾等分
        all_factor_df_dict (dict): 將所有因子資料的DF存在一個dict當中，例如: all_factor_df_dict = {'roe': roe_data_df, 'pb': pb_data_df, ...}
    Returns:
        factor_sum_df_interpolated (dict): 雙因子內插值相加後的加權總分
    Function:
        該因子選股的方法是根據台股研究室的內插法
        計算多個因子內插值的加權總分，如果有任一因子為nan，其他因子不為nan，則加總也是nan
        最後根據因子切割的大小quantile，回傳該權重的position
    """
    # 取得個因子名稱
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    factor_df_dict = {}
    # 判斷雙因子是否相同
    # 照理來雙因子選股不該帶入兩個相同的因子
    if factor_1 == factor_2:
        factor_df_dict[factor_1] = all_factor_df_dict[factor_1]
        # 將第二個因子KEY值做出名稱差異
        factor_2 = factor_2 + "'"
        factor_df_dict[factor_2] = all_factor_df_dict[factor_1]
        # 比重、排序也要加上第二個重複因子的值\
        factor_ratio_dict[factor_2] = factor_ratio_dict[factor_1]
        factor_asc_dict[factor_2] = factor_asc_dict[factor_1]
    else:
        factor_df_dict[factor_1] = all_factor_df_dict[factor_1]
        factor_df_dict[factor_2] = all_factor_df_dict[factor_2]

    # 計算因子DF的內插值
    # 初始化一個空字典以存儲插值後的數據框
    factor_df_interpolated = {}
    # 遍歷 factor_df_dict 中的每一個鍵值對
    for name, df in factor_df_dict.items():
        # 從 factor_asc_dict 中取得相應的因子，如果未找到則默認為 False
        factor = factor_asc_dict.get(name, False)
        # 調用 cal_interpolated_of_df 函數，傳入當前的數據框和因子
        # 呼叫計算單因子評分的方法，計算方式可分為: 內插值(interpolated)、排名(ranked)
        interpolated_df = cal_interpolated_of_df(df, factor,method)
        # 將結果添加到 factor_df_interpolated 字典中
        factor_df_interpolated[name] = interpolated_df

 
    # 將每個因子的內插值乘上對應的比重
    factor_interpolated_weighted = {
        name: interpolated * factor_ratio_dict[name]
        for name, interpolated in factor_df_interpolated.items()
    }

    # 將所有因子的加權內插值相加，得加權總分，並轉成CustomDataFrame
    factor_sum_df_interpolated = CustomDataFrame(
        sum(factor_interpolated_weighted.values())
    )

    # 回傳多因子權重加總後的dataframe
    return factor_sum_df_interpolated.divide_slice(quantile)


# 雙因子切割(過濾篩選多因子)
def factor_analysis_two_factor_AA(
    factor_name_list: list,
    factor_asc_dict: dict,
    quantile: int = 4,
    all_factor_df_dict: dict = None,
) -> dict:
    """
    實現 Achieving Alpha 的雙因子選股方法(過濾篩選)，
    強調第一個因子，減弱第二個因子的影響。

    Args:
        factor_name_list (list): 包含多個因子名稱的列表（例如，['roe', 'pb']）。
        factor_asc_dict (dict): 包含多個因子排序方向的字典。
        quantile (positive-int): 進行因子切割的分位數。
        all_factor_df_dict (dict): 包含所有因子資料框的字典
                                  （例如，{'roe': roe_data_df, 'pb': pb_data_df, ...}）。

    Returns:
        dict: 包含每個分位數的持倉的字典。

    """

    # 取得個因子名稱()
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = CustomDataFrame(all_factor_df_dict[factor_1])
    factor_2_df = CustomDataFrame(all_factor_df_dict[factor_2])
    # 從Input擷取個因子的排序方向
    factor_1_asc = factor_asc_dict[factor_1]
    factor_2_asc = factor_asc_dict[factor_2]
    # 先將第一個因子根據quantile值做切割
    factor_1_slice_dict = factor_1_df.divide_slice(quantile, factor_1_asc)
    # 先進行MASK處理
    factor1_mask_factor2 = {}
    for q, df in factor_1_slice_dict.items():
        # key = 'Quantile_1_MASK_factor2'
        key = f"{q}_MASK_factor2"
        value = MASK(df, factor_2_df)
        factor1_mask_factor2[key] = value

    result = {}
    for i in range(quantile):
        # key = f"Quantile{i+1}_{factor_1}_{factor_2}"
        key = f"Quantile_{i+1}"
        tmp_str = "Quantile_" + str(i + 1) + "_MASK_factor2"
        tmp_list = factor1_mask_factor2[tmp_str].divide_slice(quantile, factor_2_asc)
        result[key] = tmp_list["Quantile_" + str(i + 1)]
    return result


# 雙因子切割(直接做AND交集運算)
def factor_analysis_two_factor_AND(
    factor_name_list: list,
    factor_asc_dict: dict,
    quantile: int = 4,
    all_factor_df_dict: dict = None,
) -> dict:
    """將兩個因子DF經divide_slice後，根據Quantile 執行AND運算

    Args:
        factor_name_list (list): 包含多個因子名稱的列表（例如，['roe', 'pb']）。
        factor_asc_dict (dict): 包含多個因子排序方向的字典。
        quantile (positive-int): 進行因子切割的分位數。
        all_factor_df_dict (dict): 包含所有因子資料框的字典
                                  （例如，{'roe': roe_data_df, 'pb': pb_data_df, ...}）。

    Returns:
        dict: 包含每個分位數的持倉的字典。

    """

    # 取得個因子名稱
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = all_factor_df_dict[factor_1]
    factor_2_df = all_factor_df_dict[factor_2]
    # 從Input擷取個因子的排序方向
    factor_1_asc = factor_asc_dict[factor_1]
    factor_2_asc = factor_asc_dict[factor_2]

    factor_1_after_slice = factor_1_df.divide_slice(quantile, factor_1_asc)
    factor_2_after_slice = factor_2_df.divide_slice(quantile, factor_2_asc)
    # print(factor_1_after_slice)
    result = {}
    for i in range(quantile):
        # key = f"Quantile{i+1}_{factor_1}_{factor_2}"
        key = f"Quantile_{i+1}"
        value = (
            factor_1_after_slice["Quantile_" + str(i + 1)]
            & factor_2_after_slice["Quantile_" + str(i + 1)]
        )
        result[key] = value

    return result


# 單因子切割
def factor_analysis_single(factor_df, factor_asc: bool, quantile: int = 4) -> dict:
    """
    單因子根據值的大小與排序方向做分割

    Args:
        factor_df (dataframe): 單一因子的資料
        factor_asc (bool): 排序的方向，F:越大越好; T:越小越好
        quantile (positive-int): 打算將因子切割成幾等分

    Returns:
        各分位的position，回傳一個包含多個df的dict
    """

    return factor_df.divide_slice(quantile, factor_asc)


 
 

if __name__ == "__main__":
    pass
 

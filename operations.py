import pandas as pd
from dataframe import CustomDataFrame
from datetime import datetime, timedelta

# 外部import
from itertools import combinations
from typing import List
import pandas as pd
import numpy as np
from itertools import combinations


# 計算排列組合(雙因子會用到)
def generate_combinations(arr: List):
    """
    生成給定列表的所有可能的兩兩排列組合，。

    Args:
        arr (List): 輸入的列表。

    Returns:
        List: 包含所有兩兩排列組合的列表。

    Example:
        my_array = ['A', 'B', 'C']
        result_combinations = generate_combinations(my_array)
        print(result_combinations)
        Output: [['A', 'B'], ['A', 'C'], ['B', 'C']]

    """
    result_array = []

    # 使用 itertools 的 combinations 函數生成所有可能的兩兩組合
    for combo in combinations(arr, 2):
        result_array.append(list(combo))

    return result_array


# MASK運算(Achieving Alpha雙因子會用到)
def MASK(df_bool, df_numeric):
    """進行遮罩運算，過濾掉地一個因子沒有入選的
    Args:
        df_bool (_type_): 利用第一個因子分割後的DF
        df_numeric (_type_): 後一個因子值得DF
    Returns:
        _type_: _description_
    Exapmle:
        df_bool:
            |            | Stock 2330 | Stock 1101 | Stock 2454 | Stock 2540 |
            |------------|------------|------------|------------|------------|
            | 2021-12-31 | True       | False      | False      | True       |
            | 2022-03-31 | True       | True       | True       | False      |
            | 2022-06-30 | False      | True       | False      | False      |

        df_numeric:
            |            | Stock 2330 | Stock 1101 | Stock 2454 | Stock 2540 |
            |------------|------------|------------|------------|------------|
            | 2021-12-31 | 3          | 5          | 0          | 0.5        |
            | 2022-03-31 | 3.5        | 2.5        | 0.25       | 0          |
            | 2022-06-30 | 1.7        | 1          | 0          | 0          |

        result_df:
            |            | Stock 2330 | Stock 1101 | Stock 2454 | Stock 2540 |
            |------------|------------|------------|------------|------------|
            | 2021-12-31 | 3          | nan        | nan        | 0.5        |
            | 2022-03-31 | 3.5        | 2.5        | 0.25       | nan        |
            | 2022-06-30 | nan        | 1          | nan        | nan        |
    """
    # 使用 np.where 進行遮罩操作
    result = np.where(df_bool, df_numeric, np.nan)
    # 將結果添加到新的 DataFrame 中，並設定相同的日期索引
    result_df = pd.DataFrame(result, columns=df_numeric.columns, index=df_numeric.index)

    return CustomDataFrame(result_df)


def winsorize_row(row, winsorizing_percentile=80):
    """
    使用 Winsorizing 方法處理一行數據中的極值。

    Args:
        row (np.ndarray): 需要處理的一行數據。
        winsorizing_percentile (float): Winsorizing 的百分位數閾值，範圍在 0 到 100 之間。

    Returns:
        np.ndarray: 經過 Winsorizing 處理後的數據。

    Notes:
        Winsorizing 是通過將超過一定百分位數閾值的值替換為該閾值來處理極值，而不是直接刪除。

    Example:
        row_data = np.array([1, 2, 3, 100, 5, 6, 7])
        winsorized_row = winsorize_row(row_data, winsorizing_percentile=80)
        print(winsorized_row)
        # Output: [ 2.  2.  3.  7.  5.  6.  7.]

    """
    # 設定 Winsorizing 的百分位數閾值
    # 計算 Winsorizing 的閾值
    # 下面計算百分比的地方本來是用np.percentile(無法處理含空值資料)
    lower_bound = np.nanpercentile(row, (100 - winsorizing_percentile) / 2)
    upper_bound = np.nanpercentile(
        row, winsorizing_percentile + (100 - winsorizing_percentile) / 2
    )

    # 將超過閾值的值替換為閾值
    row_winsorized = np.clip(row, lower_bound, upper_bound)

    return row_winsorized


def remove_outliers_tukey(series, factor=1.5):
    """
    Args:
        series (pd.Series): 要處理的 pandas Series
        factor (float): Tukey's Fences 的濾波因子，默認為 1.5
    Returns:
        series_no_outliers (pd.Series): 濾波後的 Series
    Function:
        使用 Tukey's Fences 濾波器移除極端值
    """
    # 計算四分位數
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    # 計算 IQR
    IQR = Q3 - Q1

    # 計算 Tukey's Fences 的上下界
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # 濾掉超出界限的數據
    series_no_outliers = series[(series >= lower_bound) & (series <= upper_bound)]

    return series_no_outliers


# 雙因子的切割
# 計算dataframe的內插值(加權法用)
def cal_interpolated_of_df(
    df,
    ascending: bool = False,
    method: str = "ranked",
    handle_outliers: bool = True,
):
    """
    這邊有兩個主要參數，是否去極值，計算因子分數的方法，內插或排名
    Args:
        df (dataframe): 想要計算內插值的資料
        ascending (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        method(str): 使用直接內差或是排名計算法
        handle_outliers(bool): 判斷是否要處理每一列的極端值
    Returns:
        interpolated_df(dataframe): 計算完因子分數後的dataframe
    Function:
        以ROW為基準，計算每一ROW的內插值，最大1最小0
        內插值 = (當前值 - 最小值) / (最大值 - 最小值)
    """
    # 呼叫winsorize_row執行去極值
    if handle_outliers:
        # 對每一列應用 Winsorizing 函數
        df = df.apply(remove_outliers_tukey, axis=1, factor=1.5)
    # 判斷計算內差值的方法
    if method == "ranked":
        # 根據因子的ascending做進一步處理
        # 這邊判斷ASC的邏輯相反
        ranked_factor_df = df.rank(
            axis=1, ascending=not ascending, method="average", na_option="keep"
        )
        # 定義分數區間
        bin_interval = 100
        # 計算每個分數區間的分數
        ranked_factor_df_after_bin = np.ceil(ranked_factor_df / bin_interval)
        return ranked_factor_df_after_bin
    elif method == "interpolated":
        # 計算每行的最大值和最小值
        max_values = df.max(axis=1)
        min_values = df.min(axis=1)
        # 計算內插值
        interpolated_df = (df.sub(min_values, axis=0)).div(
            (max_values - min_values), axis=0
        )
        # 根據因子的ascending做進一步處理
        if ascending:
            return 1 - interpolated_df
        else:
            return interpolated_df


def generate_combinations(arr):
    """
    生成給定陣列的所有唯一兩兩元素的排列組合。

    Args:
    - arr (list): 一個包含元素的列表，用於生成排列組合。

    Returns:
    list: 一個列表，包含從輸入陣列中生成的所有唯一兩兩元素的排列組合。
          每個排列組合都表示為一個列表。

    Example: 
    >>> my_array = ['A', 'B', 'C']
    >>> result_combinations = generate_combinations(my_array)
    >>> print(result_combinations)
    [['A', 'B'], ['A', 'C'], ['B', 'C']]
    """
    result_array = []

    # 使用 itertools 的 combinations 函數生成所有可能的兩兩組合
    for combo in combinations(arr, 2):
        result_array.append(list(combo))

    return result_array


# if __name__ == "__main__":

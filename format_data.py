# 從原本get_data.py節取出來的，更模組化
# 這些function是將原始DB TABLE的資料轉成get() api可用格式

import pandas as pd
from dataframe import CustomDataFrame
from datetime import datetime, timedelta

# # Abstract API：
# from talib import abstract


def format_price_data(raw_price_data, item):
    """
    格式化原始價格資料成樞紐表格式。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。
    item (str)：要提取的項目（例如："open"、"high"、"low"、"close"、"volume"、"market_capital"）。

    返回：
    DataFrame：包含每家公司和日期的項目值的樞紐表。
    """
    selected_data = raw_price_data[["date", item, "company_symbol"]]
    pivot_data = selected_data.pivot_table(
        index="date", columns="company_symbol", values=item
    )
    return CustomDataFrame(pivot_data)


def format_report_data(raw_report_data, factor):
    """
    格式化原始報告資料成不同因子的DataFrame字典。

    參數：
    raw_report_data (DataFrame)：來自DB表格的原始報告資料。
    factor (str)：要提取的因子名稱。

    返回：
    dict：包含每家公司和日期的因子值的DataFrame字典。
    """
    # print(raw_report_data.duplicated(['date', 'company_symbol']))
    raw_report_data = raw_report_data.drop_duplicates(subset=['date', 'company_symbol', 'factor_name'])
    unique_ids = raw_report_data["factor_name"].unique()
    # print(unique_ids)
    dfs_by_id = {}
    for unique_id in unique_ids:
        temp_df = raw_report_data[raw_report_data["factor_name"] == unique_id].pivot(
            index="date", columns="company_symbol", values="factor_value"
        )
        dfs_by_id[unique_id] = CustomDataFrame(temp_df)

    return dfs_by_id[factor]

def format_report_fundamentals_data(raw_report_fundamentals, factor):
    #raw_report_fundamentals = raw_report_fundamentals.drop_duplicates(subset=['date', 'company_symbol', 'factor_name'])

    unique_ids = raw_report_fundamentals["report_fundamentals_name"].unique()
    dfs_by_id = {}
    for unique_id in unique_ids:
        temp_df = raw_report_fundamentals[raw_report_fundamentals["report_fundamentals_name"] == unique_id].pivot(
            index="date", columns="company_symbol", values="report_fundamentals_value"
        )
        dfs_by_id[unique_id] = CustomDataFrame(temp_df)

    return dfs_by_id[factor]


def handle_price_data(raw_price_data):
    """
    格式化原始價格資料成包含不同價格項目的字典。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。

    返回：
    dict：包含"open"、"high"、"low"、"close"、"volume"和"market_capital"的DataFrame的字典。
    """
    all_open = format_price_data(raw_price_data, "open")
    all_high = format_price_data(raw_price_data, "high")
    all_low = format_price_data(raw_price_data, "low")
    all_close = format_price_data(raw_price_data, "close")
    all_volume = format_price_data(raw_price_data, "volume")
    all_market_capital = format_price_data(raw_price_data, "market_capital")

    all_daily_pe = format_price_data(raw_price_data, "daily_pe")

    all_price_dict = {
        "open": all_open,
        "high": all_high,
        "low": all_low,
        "close": all_close,
        "volume": all_volume,
        "market_capital": all_market_capital,
        "daily_pe": all_daily_pe
    }
    return all_price_dict


def get_each_company_daily_price(raw_price_data, company_symbol):
    """
    獲取特定公司的每日價格資料。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。
    company_symbol (str)：要檢索資料的公司代號。

    返回：
    DataFrame：指定公司的每日價格資料。
    """
    filtered_df = raw_price_data[raw_price_data["company_symbol"] == company_symbol]
    filtered_df.set_index("date", inplace=True)
    filtered_df = filtered_df.sort_index(ascending=True)
    return filtered_df


def get_all_company_symbol(raw_price_data):
    """
    獲取原始價格資料中所有獨特的公司代號清單。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。

    返回：
    list：獨特的公司代號清單。
    """
    unique_symbols = raw_price_data["company_symbol"].drop_duplicates()
    unique_symbols_list = unique_symbols.tolist()
    return unique_symbols_list


def get_number_of_indicator_return(indname, tmp_company_daily_price):
    """
    確定指標會回傳多少個DataFrame或單一值。

    Args:
    indname (str): 要計算的指標名稱。
    tmp_company_daily_price (pandas.DataFrame): 包含一間公司每日價格資料的DataFrame。

    Returns:
    int: 回傳值的數量，如果該指標回傳多個DataFrame，則為DataFrame的欄位數量；如果只回傳單一值，則為1。

    註解:
    - 此函數用於確定指標的回傳值數量，以便在後續處理中進行適當的資料處理。
    - 函數通過呼叫指定的指標函數（`indname`）來取得結果，然後檢查結果的型別來確定回傳值的數量。
    - 如果結果是DataFrame，則表示有多個回傳值，回傳其欄位數量。
    - 如果結果是Series，則表示只有單一回傳值，回傳1。
    """
    # 先根據該指標會回傳幾個DF來宣告
    # 隨便帶入一間公司做運算
    # tmp_company_daily_price = get_each_company_daily_price(self.raw_price_data, company_symbol)
    tmp_result = eval("abstract." + indname + "(tmp_company_daily_price)")
    if isinstance(tmp_result, pd.core.frame.DataFrame):
        # 如果是DataFrame，表示有多個回傳值
        num_of_return = tmp_result.shape[1]
        # print("回傳數量: ",num_of_return)
        return num_of_return
    elif isinstance(tmp_result, pd.Series):
        # 如果是Series，表示只有一個回傳值
        # 直接將該回傳值作為單一元素回傳
        # print("回傳數量: ",1)
        return 1


def adjust_index_of_report(df):
    # 將日期字符串轉換為 datetime 對象
    df.index = pd.to_datetime(df.index)

    # 創建一個新的索引列表
    new_index = []

    # 遍歷原始 DataFrame 的索引
    for current_date in df.index:
        # 根據規則進行日期調整
        if current_date.month == 3:
            new_date = current_date.replace(month=5, day=15)
        elif current_date.month == 6:
            new_date = current_date.replace(month=8, day=31)
        elif current_date.month == 9:
            new_date = current_date.replace(month=11, day=15)
        elif current_date.month == 12:
            new_date = (current_date + timedelta(days=90)).replace(day=31)
        else:
            # 如果日期不符合規則，保持不變
            new_date = current_date

        # 將新日期添加到新索引列表
        new_index.append(new_date)

    # 將新索引設置為 DataFrame 的索引
    df.index = new_index

    return df


# if __name__ == "__main__":

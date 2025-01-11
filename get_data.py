from database import Database
from dataframe import CustomDataFrame
from format_data import *
# import talib
import pandas as pd
import numpy as np
from factors.registry import FactorRegistry
from factor_cache import MemoryCache
from parallel_calculator import HybridParallelCalculator
from typing import List, Dict
import logging
from multiprocessing import cpu_count
# # Abstract API：

# import sys
# sys.path.insert(0, 'C:/Users/user/Documents/iplab/stock_factor_lab')

def get_input_args(attr):
    input_names = attr.input_names
    refine_input_names = []
    for key, val in input_names.items():
        if "price" in key:
            if isinstance(val, list):
                refine_input_names += val
            elif isinstance(val, str):
                refine_input_names.append(val)

    return refine_input_names


class Data:
    def __init__(self):
        """
        初始化物件，連接資料庫並下載相關資料。

        Args:
            self: 類的實例（通常是類的物件，不需要額外指定）。
        """
        # 連接資料庫並下載股價資料
        self.db = Database()
        self.raw_price_data = self.db.get_daily_stock()
        self.all_price_dict = self.handle_price_data()
        # 下載財報資料
        self.raw_report_fundamentals_data = self.db.get_report_fundamentals()
        self.logger = logging.getLogger('Data')
        self.logger.setLevel(logging.INFO)

        # self.raw_report_data = self.db.get_finance_report()
        # self.raw_taiex_data = self.db.get_taiex_data()
        
        #------- 使用一半的記憶體 ------#
        self.optimal_processes = cpu_count() - 2  # 預留兩個核心給系統使用
        self.memory_cache = MemoryCache(max_cache_size_gb=48)  

    def format_price_data(self, item):
        """
        從原始股價資料中處理特定項目的資料。

        Args:
            item (str): 要處理的項目名稱。

        Returns:
            pandas.DataFrame: 處理後的股價資料的DataFrame。
        """
        return format_price_data(self.raw_price_data, item)
    
    def format_report_fundamentals_data(self, factor):

        return format_report_fundamentals_data(self.raw_report_fundamentals_data, factor)

    def format_report_data(self, factor):
        """
        從原始財報資料中處理特定因子的資料。

        Args:
            factor (str): 要處理的財報因子名稱。

        Returns:
            pandas.DataFrame: 處理後的財報資料的DataFrame。
        """
        return format_report_data(self.raw_report_data, factor)

    def handle_price_data(self):
        """
        處理原始股價資料並返回一個字典。

        Returns:
            dict: 包含處理後的股價資料的字典。
        """
        return handle_price_data(self.raw_price_data)

    # 取得資料起點
    def get(self, dataset):
        """
        從資料庫中取得資料並返回相應的DataFrame。

        Args:
            self: 類的實例（通常是類的物件，不需要額外指定）。
            dataset (str): 想要的資料名稱，格式如 "price:close" 或 "report:roe"。

        Returns:
            pandas.DataFrame or dict: 一個包含所有公司股價/財報相關資料的DataFrame 或一個包含多個財報資料的字典。

        註解:
        - 此方法根據輸入的資料名稱（dataset）擷取相對應的資料。
        - 如果資料名稱是 "price"，則返回股價相關的DataFrame。
        - 如果資料名稱是 "report"，則返回多個財報資料的字典，每個財報資料對應一個鍵。
        - 資料名稱應以冒號分隔，例如 "price:close" 或 "report:roe"。
        - 如果輸入格式不正確，將列印錯誤訊息。

        使用示例:
        

        # 創建類的實例
        my_instance = YourClass()
        # 呼叫get方法取得股價資料
        price_data = my_instance.get("price:close")
        # price_data 可能是一個DataFrame，包含股價相關資料。
        # 或者呼叫get方法取得多個財報資料
        report_data = my_instance.get("report:roe,eps")
        # report_data 是一個字典，包含多個財報資料，例如 {"roe": DataFrame, "eps": DataFrame}。
        

        """
        # 使用 lower() 方法將字串轉換為小寫
        dataset = dataset.lower()
        dataset = dataset.replace(" ", "")
        # 使用 split() 函數按 ":" 分隔字串
        parts = dataset.split(":")
        if len(parts) == 2:
            subject = parts[0]
            item = parts[1]
        else:
            print("輸入格式錯誤(Ex. price:close)")

        if subject == "price":
            # 呼叫處理開高低收的FUNCT
            price_data = self.all_price_dict[item]
            # 處理價量資料缺漏植問題: ffill
            price_data.fillna(method="ffill", inplace=True)
            price_data.index = pd.to_datetime(price_data.index)
            return price_data
        elif subject == "report":
            # 財報資料的Header為大寫
            item = item.upper().replace(" ", "")
            report_data = self.format_report_data(item)
            # 財報資料日期調整
            adjusted_report_data = adjust_index_of_report(report_data)
            # 處理財報資料缺漏植問題: ffill
            adjusted_report_data.fillna(method="ffill", inplace=True)
            # return report_data
            return adjusted_report_data
        
        elif subject == "report_fundamentals":
            # 財報資料的 Header 為大寫
            item = item.upper().replace(" ", "")
            report_fundamentals_data = self.format_report_fundamentals_data(item)
            # 財報資料日期調整
            adjusted_report_fundamentals_data = adjust_index_of_report(report_fundamentals_data)
            # 處理財報資料缺漏植問題: ffill
            adjusted_report_fundamentals_data.fillna(method="ffill", inplace=True)
            # return report_data
            return adjusted_report_fundamentals_data

        # EX. 想要取得台股加權指數(收盤價):
        # taiex_close = data.get("taiex: close")
        elif subject == "taiex":
            self.raw_taiex_data = self.db.get_taiex_data() # 大盤只會顯示一次的問題
            self.raw_taiex_data.set_index("date", inplace=True)
            self.raw_taiex_data.index = pd.to_datetime(self.raw_taiex_data.index)
            return self.raw_taiex_data[[item]]

        else:
            print("目前資料來源有price、report、taiex")

    async def get_factors_with_hybrid_parallel(
        self,
        factor_names: List[str],
        use_cache: bool = True,
        force_refresh: bool = False,
        max_age_minutes: int = 480,
        n_processes: int = None,
        n_threads: int = None,
        bband_params: dict = None
    ) -> Dict:
        """
        使用混合並行計算方式處理多個因子計算
        
        參數:
        factor_names: 要計算的因子名稱列表
        use_cache: 是否使用快取
        force_refresh: 是否強制重新計算
        max_age_minutes: 快取有效時間（分鐘）
        n_processes: 處理程序數量
        bband_params: 布林通道參數設定

        返回:
        Dict: 因子計算結果的字典
        """
        if n_processes is None:
            n_processes = cpu_count() - 2

        calculator = HybridParallelCalculator(
            data=self,
            n_processes=n_processes,
            n_threads=n_threads
        )

        try:
            return await calculator.calculate_factors_hybrid(
                factor_names=factor_names,
                use_cache=use_cache,
                force_refresh=force_refresh,
                max_age_minutes=max_age_minutes,
                bband_params=bband_params
            )
        except Exception as e:
            calculator.logger.error(f"因子計算過程發生錯誤: {str(e)}")
            raise

    def indicator(self, indname, adjust_price=False, resample="D", **kwargs):
        """支援 Talib 和 pandas_ta 上百種技術指標，計算 2000 檔股票、10年的所有資訊。

        在使用這個函式前，需要安裝計算技術指標的 Packages

        * [Ta-Lib](https://github.com/mrjbq7/ta-lib)
        * [Pandas-ta](https://github.com/twopirllc/pandas-ta)

        Args:
            indname (str): 指標名稱，
                以 TA-Lib 舉例，例如 SMA, STOCH, RSI 等，可以參考 [talib 文件](https://mrjbq7.github.io/ta-lib/doc_index.html)。

                以 Pandas-ta 舉例，例如 supertrend, ssf 等，可以參考 [Pandas-ta 文件](https://twopirllc.github.io/pandas-ta/#indicators-by-category)。
            adjust_price (bool): 是否使用還原股價計算。
            resample (str): 技術指標價格週期，ex: D 代表日線, W 代表週線, M 代表月線。
            market (str): 市場選擇，ex: TW_STOCK 代表台股, US_STOCK 代表美股。
            **kwargs (dict): 技術指標的參數設定，TA-Lib 中的 RSI 為例，調整項為計算週期 `timeperiod=14`。
        建議使用者可以先參考以下範例，並且搭配 talib官方文件，就可以掌握製作技術指標的方法了。
        """
        package = None

        try:
            from talib import abstract
            import talib

            attr = getattr(abstract, indname)
            # print("計算指標:", attr)
            package = "talib"
        except:
            try:
                import pandas_ta

                # test df.ta has attribute
                getattr(pd.DataFrame().ta, indname)
                attr = lambda df, **kwargs: getattr(df.ta, indname)(**kwargs)
                package = "pandas_ta"
            except:
                raise Exception("Please install TA-Lib or pandas_ta to get indicators.")

        # market = get_market_info(user_market_info=market)

        # close = market.get_price('close', adj=adjust_price)
        # open_ = market.get_price('open', adj=adjust_price)
        # high = market.get_price('high', adj=adjust_price)
        # low = market.get_price('low', adj=adjust_price)
        # volume = market.get_price('volume', adj=adjust_price)

        close = self.get("price:close")
        open_ = self.get("price:open")
        high = self.get("price:high")
        low = self.get("price:low")
        volume = self.get("price:volume")

        close.index = pd.to_datetime(close.index)
        open_.index = pd.to_datetime(open_.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        volume.index = pd.to_datetime(volume.index)

        if resample.upper() != "D":
            close = close.resample(resample).last()
            open_ = open_.resample(resample).first()
            high = high.resample(resample).max()
            low = low.resample(resample).min()
            volume = volume.resample(resample).sum()

        dfs = {}
        default_output_columns = None
        for key in close.columns:
            # 組成單一公司的開高低收量
            # 這邊也跟計算效能有關

            prices = {
                "open": open_[key].ffill(),
                "high": high[key].ffill(),
                "low": low[key].ffill(),
                "close": close[key].ffill(),
                "volume": volume[key].ffill(),
            }

            if package == "pandas_ta":
                prices = pd.DataFrame(prices)
                s = attr(prices, **kwargs)

            elif package == "talib":
                # abstract_input 存放該指標需要開/高/低/收哪幾項
                # 以idname='AD'為例，abstract_input = ['high', 'low', 'close', 'volume']
                abstract_input = list(attr.input_names.values())[0]
                abstract_input = get_input_args(attr)

                # quick fix talib bug
                if indname == "OBV":
                    abstract_input = ["close", "volume"]

                if indname == "BETA":
                    abstract_input = ["high", "low"]

                if isinstance(abstract_input, str):
                    abstract_input = [abstract_input]
                # paras 存放實際股價資料的list['close','high']
                paras = [prices[k] for k in abstract_input]

                # 實際呼叫talib計算的地方
                # attr屬於一種talib._ta_lib.Function
                # 固可以在後面加(參數)呼叫該方法
                # 一個s表示一間公司
                # type(s): dict/ndarray
                s = attr(*paras, **kwargs)

            else:
                raise Exception("Cannot determine technical package from indname")

            # 根據回傳的資料格式，搭配不同的處理方法
            # 最後要統一格式成dict
            # talib指標回傳一個值屬於nparray,多個值屬於list
            if isinstance(s, list):
                # 例如: BBANDS、MACD、STOCH
                # print("result of cal is: list")
                s = {i: series for i, series in enumerate(s)}

            if isinstance(s, np.ndarray):
                # 例如: ADX、RSI、MA5
                # print("result of cal is: ndarray")
                # 將np.ndarrat轉成dict
                s = {0: s}

            if isinstance(s, pd.Series):
                # print("result of cal is: Series")
                s = {0: s.values}

            if isinstance(s, pd.DataFrame):
                # print("result of cal is: DataFrame")
                s = {i: series.values for i, series in s.items()}

            if default_output_columns is None:
                default_output_columns = list(s.keys())

            # dfs是一個dict結構，以BBANDS為例
            # 迴圈會每次逐行加上公司column
            # dfs[0]存放所有公司的upper
            # dfs[1]存放所有公司的middle
            for colname, series in s.items():
                if colname not in dfs:
                    dfs[colname] = {}
                dfs[colname][key] = series if isinstance(series, pd.Series) else series

            # print("dfs:", dfs)

        newdic = {}
        for key, df in dfs.items():
            # 原本的計算結果都存放在dict中，最後在一次轉換成dataframe
            newdic[key] = pd.DataFrame(df, index=close.index)

        ret = [newdic[n] for n in default_output_columns]
        ret = [d.apply(lambda s: pd.to_numeric(s, errors="coerce")) for d in ret]

        if len(ret) == 1:
            return CustomDataFrame(ret[0])

        # 回傳tuple
        return tuple([CustomDataFrame(df) for df in ret])


if __name__ == "__main__":
    data = Data()
    vmo_result = data.get_indicator('vmo')
    # 測試輸出股價資料
    # close = data.get("price:close")
    # print("收盤價:", close)
    # 測試輸出財報資料
    # roe = data.get("report:roe, EPS")
    # print(roe)
    # rsi = data.indicator('RSI', timeperiod=14)
    # price = d
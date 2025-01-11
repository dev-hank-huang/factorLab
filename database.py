import configparser
from utils.config import Config
from dataframe import CustomDataFrame

import pymysql
import pandas as pd


class Database:
    def __init__(self):
        self._config = Config()
        self._db_data= self._config.get_database_config()
        # self.connection()

    """
    stock_index = {open, high, low, close, volume, market_capital}
    """

    # 建立與DB的連線
    def create_connection(self):
        # 檢查DB版本&連線成功
        try:
            config_host = self._db_data["host"]
            config_port = int(self._db_data["port"])
            config_user = self._db_data["user"]
            config_password = self._db_data["password"]
            config_db = self._db_data["db"]
            config_charset = self._db_data["charset"]

            db = pymysql.connect(
                host=config_host,
                port=config_port,
                user=config_user,
                passwd=config_password,
                db=config_db,
                charset=config_charset,
            )
            # print("連線成功")
            return db
        except Exception as e:
            print(e)
            print("無法連結資料庫")
            return e

    # 取的公司的開高低收(stock)
    def get_daily_stock(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            # data = cursor.fetchone()
            # print('連線成功')

            # 選取台股(有帶入一些條件，避免數量過多)
            sql = " SELECT company_symbol,name,date,open,high,low,close,volume,market_capital, daily_pe \
                    FROM company RIGHT JOIN stock ON company.id = stock.company_id \
                    WHERE exchange_name='TWSE' "
                    # AND date > 2018-01-01"
            # AND company_symbol>8700
            # AND company_symbol<9000"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = [
                "company_symbol",
                "name",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "market_capital",
                "daily_pe",
            ]
            df = CustomDataFrame(data, columns=columns)
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e

    # 取得公司的財報(factorvalue)
    # def get_finance_report(self):
    #     try:
    #         db = self.create_connection()
    #         cursor = db.cursor()
    #         sql = " SELECT date, company_symbol, factor_name, factor_value \
    #                 FROM factor RIGHT JOIN factorvalue ON factor.id = factorvalue.factor_id  \
    #                 LEFT JOIN  company ON factorvalue.company_id = company.id \
    #                 WHERE exchange_name='TWSE'\
    #                 AND date > 2018-01-01"
    #         # AND company_symbol>8700
    #         # AND company_symbol<9000"

    #         cursor.execute(sql)
    #         data = cursor.fetchall()
    #         columns = ["date", "company_symbol", "factor_name", "factor_value"]
    #         df = CustomDataFrame(data, columns=columns)
    #         # print('The raw data get from database:\n')
    #         # print(df)
    #         return df

    #     except Exception as e:
    #         print(e)
    #         print("無法執行SQL語法")
    #         return e
        
    def get_report_fundamentals(self):
        try:
            db = self.create_connection()
            cursor = db.cursor()
            sql = " SELECT date, company_symbol, tw_indicator, indicator_value \
                    FROM singleindicator RIGHT JOIN financialreportindicatorvalue ON singleindicator.id = financialreportindicatorvalue.indicator_id  \
                    LEFT JOIN  company ON financialreportindicatorvalue.company_id = company.id \
                    WHERE exchange_name='TWSE'"

            cursor.execute(sql)
            data = cursor.fetchall()
            columns = ["date", "company_symbol", "report_fundamentals_name", "report_fundamentals_value"]
            df = CustomDataFrame(data, columns=columns)
            # print('The raw data get from database:\n',df)
            # print(df)
            return df

        except Exception as e:
            print(e)
            print("無法執行SQL語法")
            return e

    # 取得台股加權指數(taiex)
    # def get_taiex_data(self):
    #     try:
    #         db = self.create_connection()
    #         cursor = db.cursor()
    #         # sql = " SELECT date, open, high, low, close, volume, market_capital FROM taiex"
    #         sql = " SELECT date, open, high, low, close, volume FROM taiex"

    #         cursor.execute(sql)
    #         data = cursor.fetchall()    
    #         # columns = ["date", "open", "high", "low", "close", "volume", "market_capital"]
    #         columns = ["date", "open", "high", "low", "close", "volume"]

    #         df = CustomDataFrame(data, columns=columns)
    #         # print('The raw data get from database:\n')
    #         # print(df)
    #         return df

    #     except Exception as e:
    #         print(e)
    #         print("無法執行SQL語法")
    #         return e
if __name__ == "__main__":
    db = Database()
    # db.create_connection()

    # daily = db.get_daily_stock()
    # print(daily)
    # # # daily.to_csv('./OutputFile/daily.csv')

    # db.get_report_fundamentals()
    # # # 使用布林索引過濾 DataFrame，擷取 "加工業" 種類的收盤價資料
    # # filtered_df = daily[daily["company_symbol"] == "8905"]
    # # filtered_df.set_index("date", inplace=True)
    # # # 升序排序日期索引
    # # filtered_df = filtered_df.sort_index(ascending=True)

    # # # filtered_df.to_csv("./OutputFile/filtered_df.csv")

    # # # db.select_index('open')

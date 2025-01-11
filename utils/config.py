from configparser import ConfigParser
import os

class Config():

    def __init__(self):
        self._parser = ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "../config.ini")
        # self._parser.read("../config.ini", encoding="utf-8")
        self._parser.read(config_path, encoding="utf-8")

    def get_database_config(self):
        database_config = {}
        if self._parser.has_section("database"):
            database_config["host"]     = str(self._parser["database"]["host"])
            database_config["port"]     = int(self._parser["database"]["port"])
            database_config["user"]     = str(self._parser["database"]["user"])
            database_config["password"] = str(self._parser["database"]["password"])
            database_config["db"]       = str(self._parser["database"]["db"])
            database_config["charset"]  = str(self._parser["database"]["charset"])
            # print("database config: ", database_config)
        else:
            # 如果配置文件中沒有[database]部分，你可以處理這個情況，例如設置默認值或者引發錯誤
            raise KeyError("No 'database' section found in config.ini")
        
        return database_config
    
# if __name__ == "__main__":
#     config = Config()
#     config.get_database_config()
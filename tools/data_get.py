import backtrader as bt
import tushare as ts
import datetime
import pandas as pd
from tools.db_mysql import get_engine, get_enginebitcoin
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
class DataGet:
    _data_cache={}
    _db_access_count=0
    @staticmethod
    def _get_cache_key(code,period):
        return f"{code}_{period}"
    
    @staticmethod
    def _load_from_cache(cache_key):
        try:
            cache_dir = "./data_cache"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # 添加调试信息
            print(f"尝试从缓存加载数据: {cache_file}")
            print(f"文件是否存在: {os.path.exists(cache_file)}")
            
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    df = pickle.load(f)
                    # 确保DataFrame的索引是datetime类型
                    df.index = pd.to_datetime(df.index)
                    print(f"成功加载缓存数据，数据行数: {len(df)}")
                    DataGet._data_cache[cache_key] = df
                    return df
            return None
        except Exception as e:
            print(f"加载缓存数据时出错: {str(e)}")
            return None

    @staticmethod
    def _save_to_cache(cache_key, data):
        try:
            cache_dir = "./data_cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # 确保数据是DataFrame类型
            if not isinstance(data, pd.DataFrame):
                raise ValueError("数据必须是DataFrame类型")
            
            # 添加调试信息
            print(f"保存数据到缓存: {cache_file}")
            print(f"数据行数: {len(data)}")
            
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            DataGet._data_cache[cache_key] = data
            print("数据成功保存到缓存")
        except Exception as e:
            print(f"保存缓存数据时出错: {str(e)}")

    @staticmethod
    def get_str_to_datetime(date_str):
        """
        将日期字符串格式（yyyyMMdd、yyyyMMddHHmm、yyyyMMddHHmmss）转换为日期对象。
        :param date_str: 日期的字符串表示
        :return: 格式化后的日期字符串
        """
        # 清理输入中的非数字字符
        if isinstance(date_str, str):
            date_str = ''.join(filter(str.isdigit, date_str))
        try:
            # 如果输入已经是日期对象，则直接返回
            if isinstance(date_str, datetime.date):
                print("f{date_str} is already a date object.")
                return date_str
            date_len = len(date_str)

            if date_len == 8:  #yyyymmdd
                fmt = "%Y%m%d"
                return datetime.datetime.strptime(date_str, fmt).date()
            elif date_len == 10:  # yyyymmddHH (假设没有 mm 分钟部分)
                raise ValueError("Invalid length for time. Expected HHMM or HHMMSS.")
            elif date_len == 12:  # yyyymmddHHmm
                fmt = "%Y%m%d%H%M"
            elif date_len == 14:  # yyyymmddHHmmss
                fmt = "%Y%m%d%H%M%S"
            else:
                raise ValueError(f"Unexpected date length {date_len}. Expected 8, 12, or 14 digits.")
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError as e:
            raise ValueError(f"Invalid date format or value provided: {e}")

    @staticmethod
    def get_str_to_datetime_btc(prompt,has_time):
        """
        将日期字符串格式（yyyyMMdd、yyyyMMddHHmm、yyyyMMddHHmmss）转换为日期对象。
        :param date_str: 日期的字符串表示
        :return: 格式化后的日期字符串
        """
        while True:
            date_str = input(prompt).strip()
            # 清理输入中的非数字字符
            date_str = ''.join(filter(str.isdigit, date_str))
            if has_time == 'day':
                if not date_str.isdigit() or len(date_str) != 8:
                    print("非法输入！请确保输入为8位数字，格式为：YYYYMMDD")
                    continue
            elif has_time == 'hour':
                if not date_str.isdigit() or len(date_str) != 10:
                    print("非法输入！请确保输入为10位数字，格式为：YYYYMMDDHH")
                    continue
            elif has_time == 'min':
                if not date_str.isdigit() or len(date_str) != 12:
                    print("非法输入！请确保输入为12位数字，格式为：YYYYMMDDHHMM")
                    continue
            elif has_time == 's':
                if not date_str.isdigit() or len(date_str) != 14:
                    print("非法输入！请确保输入为14位数字，格式为：YYYYMMDDHHMMSS")
                    continue

            date_len = len(date_str)
            try:
                if date_len == 8:  #yyyymmdd
                    fmt = "%Y%m%d"
                    return datetime.datetime.strptime(date_str, fmt).date()
                elif date_len == 10:  # yyyymmddHH
                    fmt = "%Y%m%d%H"
                    return datetime.datetime.strptime(date_str, fmt)
                elif date_len == 12:  # yyyymmddHHmm
                    fmt = "%Y%m%d%H%M"
                    return datetime.datetime.strptime(date_str, fmt)
                elif date_len == 14:  # yyyymmddHHmmss
                    fmt = "%Y%m%d%H%M%S"
                    return datetime.datetime.strptime(date_str, fmt)
                else:
                    print(f"Unexpected date length {date_len}. Expected 8, 10, 12, or 14 digits.")
            except ValueError:
                print(f"无效日期: {date_str}")
                continue

    @staticmethod
    def get_date_from_int(date_str):
        """
        将日期的字符串格式（yyyyMMdd）转换为日期对象
        :param date_str: 日期的字符串表示
        :return: 转换后的日期对象
        """
        # 获取日期
        date_str = DataGet.get_str_to_datetime(date_str)
        # if has_time=='min':
        #     # 假设输入格式为 yyyyMMddHHmm
        #     date_min = date_str
        #     return date_min
        # elif has_time=='s':
        #     # 假设输入格式为 yyyyMMddHHmmss
        #     date_s = date_str
        #     return date_s
        # elif has_time=='day':
        #     # 假设输入格式为 yyyyMMdd
        #     date_full = date_str
        return date_str

    @staticmethod
    def login_ts():
        """
        登录Tushare，获取pro_api接口
        :return: 返回Tushare pro_api实例
        """
        token = 'a4ef5bd632a83a568af0497fb9a21920ada0f4d013b79685bdce16ea'
        ts.set_token(token)  # 设置Tushare Token
        pro = ts.pro_api(token)  # 获取Tushare pro接口
        return pro

    @staticmethod
    def get_data(codes, cerebro, start_date, end_date):
        """
        获取指定股票/期货代码的数据，并将其添加到回测引擎（Cerebro）中
        :param codes: 股票/期货代码（可以是单个代码或多个代码的列表）
        :param cerebro: Backtrader回测引擎实例
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        """
        pro = DataGet.login_ts()  # 登录Tushare接口
        code_list = codes if isinstance(codes, list) else codes.split()  # 确保codes是列表形式
        # 加载数据
        for code in code_list:
            #df = pro.daily(ts_code=f"{code}", start_date=start_date, end_date=end_date)  # 获取日线数据
            df = pro.daily(ts_code=f"{code}")  # 获取日线数据
            df['trade_date'] = pd.to_datetime(df['trade_date'])  # 转换交易日期为日期类型
            df.set_index('trade_date', inplace=True)  # 将交易日期设为索引
            df['openinterest'] = 0  # 初始化持仓量为0
            df = df[['open', 'high', 'low', 'close', 'vol', 'openinterest']].rename(columns={'vol': 'volume'})  # 重命名列
            df = df.sort_index()  # 按日期排序数据
            data = bt.feeds.PandasData(dataname=df)  # 转换为Backtrader的Pandas数据格式
            cerebro.adddata(data, name=code)  # 将数据添加到回测引擎中

    @staticmethod
    def get_fut_data(cerebro, codes, period):
        """获取期货数据并添加到cerebro"""
        code_list = codes if isinstance(codes, list) else [codes]
        
        for code in code_list:
            cache_key = DataGet._get_cache_key(code, period)
            
            # 首先尝试从内存缓存获取数据
            df = DataGet._data_cache.get(cache_key)
            
            if df is None:  # 如果内存中没有
                # 从pkl文件加载
                df = DataGet._load_from_cache(cache_key)
                
                if df is not None:
                    # 将数据保存到内存缓存中
                    DataGet._data_cache[cache_key] = df
                
                if df is None:  # 如果pkl文件也没有
                    # 从数据库加载
                    try:
                        connection = get_engine()
                        query = f"SELECT * FROM `{code}_{period}`"
                        df = pd.read_sql(text(query), con=connection)
                        connection.dispose()
                        
                        # 数据预处理
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.set_index('trade_date', inplace=True)
                        df['openinterest'] = 0
                        df = df[['open', 'high', 'low', 'close', 'vol', 'openinterest']].rename(columns={'vol': 'volume'})
                        df = df.sort_index()
                        
                        # 同时保存到内存缓存和文件缓存
                        DataGet._data_cache[cache_key] = df
                        DataGet._save_to_cache(cache_key, df)
                    except Exception as e:
                        print(f"从数据库加载数据时出错: {str(e)}")
                        continue
            
            try:
                data = bt.feeds.PandasData(dataname=df)
                cerebro.adddata(data, name=code)
            except Exception as e:
                print(f"添加数据到cerebro时出错: {str(e)}")

    @staticmethod
    def clear_memory_cache():
        """清理内存缓存"""
        DataGet._data_cache.clear()
        import gc
        gc.collect()

    @staticmethod
    def clear_cache():
        """清理所有缓存"""
        DataGet._data_cache.clear()
        cache_dir = "./data_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, file))
    
    ######
    # add crypto data get method
    ######
    @staticmethod
    def get_bit_data(cerebro, codes, period):
        """获取期货数据并添加到cerebro"""
        code_list = codes if isinstance(codes, list) else [codes]

        print(code_list)
        for code in code_list:
            cache_key = DataGet._get_cache_key(code, period)
            df = DataGet._load_from_cache(cache_key)

            if df is None:
                DataGet._db_access_count += 1  # 记录数据库访问次数
                # print(f"从数据库加载数据 ({DataGet._db_access_count}): {code}")

                connection = get_enginebitcoin()
                query = f"SELECT * FROM `{code}_{period}`"
                df = pd.read_sql(text(query), con=connection)


                # 数据预处理
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                df['openinterest'] = 0
                df = df[['open', 'high', 'low', 'close', 'vol', 'openinterest']].rename(columns={'vol': 'volume'})
                df = df.sort_index()
                print("正在加载数据", code)
                # 保存到缓存
                DataGet._save_to_cache(cache_key, df)
            else:
                pass
                # df.index=df.index.floor('S')
                # print(f"从缓存加载数据: {code}")

            # 转换为Backtrader数据格式并添加到cerebro
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=code)
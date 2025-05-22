import os
import sys

# 添加BackTrader目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
backtrader_dir = os.path.dirname(current_dir)
if backtrader_dir not in sys.path:
    sys.path.insert(0, backtrader_dir)  # 使用insert(0)确保优先使用本地模块


import numpy as np
import matplotlib.pyplot as plt
# from sqlalchemy import create_engine
import pandas as pd
import backtrader as bt
from tools import Log  # 使用tools包的导入
import datetime
import math
import psutil
import gc
from tools.db_mysql import get_engine


class AI_KDJ_Strategy(bt.Strategy):
    params = (
        ('backtest_start_date', None),
        ('backtest_end_date', None),
        ('atr_threshold', 0.005),  # ATR阈值，用于信号生成
        ('predict_prices', None),  # 预测价格序列
        ('predict_dates', None),   # 预测日期序列
    )

    def __init__(self):
        self.connection = get_engine()
        self._cache = None
        # 初始化数据存储
        columns = ['时间', '合约名', '信号', '单价', '手数', '总价', '手续费', '可用资金', 
                  '开仓均价', '品种浮盈', '权益', '当日收盘', '已缴纳保证金', '平仓盈亏']
        self.info = pd.DataFrame(columns=columns)
        self.info.index = range(1, len(self.info) + 1)
        # 初始化交易相关变量
        self.init_cash = 2000000
        self.cash = 2000000
        self.total_value = 0
        self.total_days = 0
        self.total_trade_time = 0
        self.win_time = 0
        self.lose_time = 0
        self.total_profit = 0
        self.total_loss = 0
        # 添加回撤相关变量
        self.max_drawdown = 0
        self.max_drawdown_percent = 0
        self.max_total_value = self.init_cash
        self.min_total_value = self.init_cash
        self.max_profit = 0
        self.max_loss = 0
        # 添加每日权益记录
        self.daily_equity = pd.DataFrame(columns=['日期', '权益', '模型名称'])
        self.daily_equity_file = 'AI_KDJ_equity.csv'
        # 初始化各类字典变量
        self.order_list = {}
        self.paper_profit = {}
        self.profit = {}
        self.log = {}
        self.average_open_cost = {}
        self.margin = {}
        self.is_trade = {}
        self.TR = {}
        self.ATR = {}
        # 初始化预测数据相关变量
        self.predict_signals = {}  # 存储预测信号
        self.current_predict_idx = 0  # 当前预测数据索引
        self.position_pct = 0.0  # 当前仓位比例
        
        # 存储所有生成的信号和交易记录
        self.all_signals = []
        self.all_positions = []
        self.all_dates = []

        for data in self.datas:
            data.cnname = self.get_margin_percent(data)['future_name']
            self.order_list[data] = [0]
            self.paper_profit[data] = 0
            self.profit[data] = 0
            self.log[data] = 0
            self.average_open_cost[data] = 0
            self.margin[data] = 0
            self.is_trade[data] = False
            # 计算ATR，使用Backtrader内置指标
            self.ATR[data] = bt.indicators.AverageTrueRange(data, period=26)
            # 如果后续仍需TR，可以单独计算： self.TR[data] = bt.indicators.TrueRange(data)

    def start(self):
        # start方法中的初始化代码已移至__init__
        pass
    
    def generate_signal(self, change, atr):
        """根据价格变化和ATR生成信号 - 与ai_backtest_interface.py一致"""
        # 改为使用绝对的ATR值而非比例
        if abs(change) < 0.005 * atr:
            return 0  # 中性
        elif 0.005 * atr <= change < 0.015 * atr:
            return 4  # 中性偏多
        elif 0.015 * atr <= change < 0.025 * atr:
            return 5  # 看多
        elif change >= 0.025 * atr:
            return 6  # 强烈看多
        elif -0.015 * atr < change <= -0.005 * atr:
            return 1  # 中性偏空
        elif -0.025 * atr < change <= -0.015 * atr:
            return 2  # 看空
        elif change <= -0.025 * atr:
            return 3  # 强烈看空
        return 0
        
    def map_signal_to_action(self, signal):
        """将信号映射为交易动作"""
        if signal == 0:
            return 0, 0.0  # 中性，不操作
        elif signal == 1:
            return 2, -0.05  # 中性偏空，做空5%
        elif signal == 2:
            return 2, -0.10  # 看空，做空10%
        elif signal == 3:
            return 2, -0.15  # 强烈看空，做空15%
        elif signal == 4:
            return 1, 0.05   # 中性偏多，做多5%
        elif signal == 5:
            return 1, 0.10   # 看多，做多10%
        elif signal == 6:
            return 1, 0.15   # 强烈看多，做多15%
            
    def next(self):
        current_date = self.datetime.date(0)
        # Log the current date of the next() method
        print(f"Strategy next() called for date: {current_date}")

        # Log before checking strategy active dates
        print(f"  Checking activation: current_date ({current_date}) vs strategy_start ({self.params.backtest_start_date}) and strategy_end ({self.params.backtest_end_date})")
        
        if self.params.backtest_start_date <= current_date <= self.params.backtest_end_date:
            print(f"  Strategy is active for {current_date}. Processing predictions.") # Log when strategy is active
            self.total_days += 1
            
            for data in self.datas:
                if data.close[0] == -1:
                    continue
                    
                # 获取当前预测数据
                if self.params.predict_prices is not None and self.current_predict_idx < len(self.params.predict_prices):
                    print(f"    Processing prediction index {self.current_predict_idx} for date {current_date}") # Log prediction processing
                    predict_price = self.params.predict_prices[self.current_predict_idx]
                    
                    # 获取上一日价格 - 可能是前一个预测或实际价格
                    if self.current_predict_idx == 0:
                        prev_price = data.close[-1]  # 第一个信号使用实际前一日收盘价
                        print(f"      Using previous actual close: {prev_price}")
                    else:
                        prev_price = self.params.predict_prices[self.current_predict_idx - 1]  # 否则使用前一天的预测价格
                        print(f"      Using previous predicted price: {prev_price}")
                    
                    # 计算变化率
                    change = (predict_price - prev_price) / prev_price
                    print(f"      Price change: {change:.4f} ({prev_price} -> {predict_price})")
                    
                    # 获取ATR并计算信号
                    atr = self.ATR[data][0]
                    atr_ratio = atr / data.close[0]  # 计算ATR相对于当前价格的比例
                    print(f"      ATR: {atr:.2f}, ATR ratio: {atr_ratio:.6f}")
                    
                    # 生成信号
                    signal = self.generate_signal(change, atr_ratio)
                    action, position_delta = self.map_signal_to_action(signal)
                    
                    # 保存信号和日期到列表
                    self.all_signals.append(signal)
                    self.all_dates.append(current_date)
                    
                    print(f"      Signal: {signal}, Action: {action}, Position delta: {position_delta}")
                    
                    # 更新仓位
                    self.position_pct += position_delta
                    self.position_pct = max(0.0, min(1.0, self.position_pct))  # 限制在 [0, 1] 范围内
                    self.all_positions.append(self.position_pct)
                    print(f"      New position: {self.position_pct:.2f}")
                    
                    # 执行交易 - 完全清空现有仓位，重新建立新仓位
                    if self.position_pct > 0:
                        self.close_all_position()  # 先平掉所有仓位
                        # 再开新仓
                        print(f"      Opening long position with size proportion: {self.position_pct:.2f}")
                        self.execute_buy(data, self.position_pct)
                    elif self.position_pct == 0:
                        print(f"      Closing all positions")
                        self.close_all_position()
                        
                    self.current_predict_idx += 1
                        
                # 更新持仓信息
                self.update_position_info(data)
                
            # 更新总权益
            self.update_total_value()
            
            # 检查爆仓
            if self.total_value <= 0:
                Log.log(f"触发爆仓条件！当前权益为负数:{self.total_value:.2f}", dt=current_date)
                self.close_all_position()
                self.env.runstop()
                return
                    
    def execute_buy(self, data, position_proportion):
        """执行买入操作"""
        fund = self.init_cash * position_proportion  # 使用比例计算
        margin_mult = self.get_margin_percent(data)
        margin_percent = margin_mult['margin']
        mult = margin_mult['mult']
        lots = int(fund / (data.close[0] * margin_percent * mult))
        if lots > 0:
            print(f"      >>> BUYING {lots} contracts at {data.close[0]}")
            self.buy(data=data, size=lots)
            self.total_trade_time += 1
        else:
            print("      >>> Not enough funds to buy any contracts")
        
    def execute_sell(self, data, position_proportion):
        """执行卖出操作"""
        fund = self.init_cash * position_proportion  # 使用比例计算
        margin_mult = self.get_margin_percent(data)
        margin_percent = margin_mult['margin']
        mult = margin_mult['mult']
        lots = int(fund / (data.close[0] * margin_percent * mult))
        if lots > 0:
            print(f"      >>> SELLING {lots} contracts at {data.close[0]}")
            self.sell(data=data, size=lots)
            self.total_trade_time += 1
        else:
            print("      >>> Not enough funds to sell any contracts")
        
    def update_position_info(self, data):
        """更新持仓信息和盈亏"""
        position = self.getposition(data)
        position_size = position.size
        
        if position_size != 0:
            # 更新持仓均价和保证金
            self.average_open_cost[data] = position.price
            margin_mult = self.get_margin_percent(data)
            margin_percent = margin_mult['margin']
            mult = margin_mult['mult']
            
            # 更新保证金
            self.margin[data] = abs(position_size) * data.close[0] * margin_percent * mult
            
            # 更新浮动盈亏
            self.paper_profit[data] = (data.close[0] - position.price) * position_size * mult if position_size > 0 else (position.price - data.close[0]) * (-position_size) * mult
            
            print(f"      Current position: {position_size} contracts, avg price: {position.price:.2f}")
            print(f"      Unrealized P&L: {self.paper_profit[data]:.2f}")
        else:
            # 清空持仓相关数据
            self.average_open_cost[data] = 0
            self.margin[data] = 0
            self.paper_profit[data] = 0
            
    def update_total_value(self):
        """更新总权益"""
        self.cash = self.broker.getcash()  # 获取当前现金
        
        total_margin = 0
        total_paper_profit = 0
        
        for data in self.datas:
            if data.close[0] == -1:
                continue
            total_margin += self.margin[data]
            total_paper_profit += self.paper_profit[data]
            
        # 计算总权益
        self.total_value = self.cash + total_paper_profit
        
        # 更新最大回撤
        if self.total_value > self.max_total_value:
            self.max_total_value = self.total_value
        if self.total_value < self.min_total_value:
            self.min_total_value = self.total_value
            
        # 计算绝对回撤和百分比回撤
        drawdown = self.max_total_value - self.total_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_percent = (drawdown / self.max_total_value) * 100
            
        # 记录日志
        current_date = self.datetime.date(0)
        print(f"      Total equity: {self.total_value:.2f}, Cash: {self.cash:.2f}, Unrealized P&L: {total_paper_profit:.2f}")
        
        # 记录每日权益
        new_row = {
            '日期': current_date,
            '权益': self.total_value,
            '模型名称': 'AI_KDJ'
        }
        self.daily_equity = pd.concat([self.daily_equity, pd.DataFrame([new_row])], ignore_index=True)
        
    def generate_signal_table(self):
        """生成信号表，类似ai_backtest_interface.py中的实现"""
        signal_to_text = {
            0: '中性',
            1: '中性偏空',
            2: '看空',
            3: '强烈看空',
            4: '中性偏多',
            5: '看多',
            6: '强烈看多'
        }
        
        # 检查是否有数据
        if not self.all_signals or not self.all_dates:
            print("没有可用的信号数据生成信号表")
            return None
            
        # 创建信号表DataFrame
        signal_df = pd.DataFrame({
            'predict_date': self.all_dates,
            'predict_price': self.params.predict_prices[:len(self.all_signals)],
            'kdj_signal': self.all_signals,
            'position': self.all_positions
        })
        
        # 添加信号文本说明
        signal_df['signal_text'] = signal_df['kdj_signal'].map(signal_to_text)
        
        return signal_df
        
    def stop(self):
        try:
            current_date = self.datetime.date(0)
        except IndexError:
            current_date = self.params.backtest_end_date
        
        # 获取当前工作目录
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "backtest_results")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n创建输出目录: {output_dir}")
        
        # 生成信号表并保存
        signal_table = self.generate_signal_table()
        if signal_table is not None:
            # 保存信号表到CSV
            signal_table_path = os.path.join(output_dir, 'AI_KDJ_signal_table.csv')
            signal_table.to_csv(signal_table_path, index=False, encoding='utf-8-sig')
            print(f"\n信号表已保存至: {signal_table_path}")
            # 打印信号表
            print("\n生成的信号表:")
            print(signal_table)
        
        # 计算回测指标
        if self.total_trade_time != 0:
            win_rate = float(self.win_time) / self.total_trade_time
            try:
                win_lose_ratio = (float(self.total_profit) / self.win_time) / (float(self.total_loss) / self.lose_time)
            except ZeroDivisionError:
                win_lose_ratio = float('inf')
                
            win_rate_2 = (1 + win_lose_ratio) * win_rate
            if math.isinf(win_rate_2):
                win_rate_2 = float('inf')
                
            # 保存回测结果
            param_info = []
            skip_attr = ['notdefault', 'isdefault']
            for name in dir(self.params):
                if not name.startswith('_') and name not in skip_attr:
                    value = getattr(self.params, name)
                    param_info.append(f'{name}={value}')
                    
            file_path = os.path.join(output_dir, '策略测试结果.csv')
            
            # 创建新的数据行
            new_row = {
                '策略名称': 'AI_KDJ',
                '时间区间': f"{self.params.backtest_start_date}-{self.params.backtest_end_date}",
                '初始资金': self.init_cash,
                '最终权益': self.total_value,
                '胜率': win_rate * 100,
                '总交易次数': self.total_trade_time,
                '盈亏比': win_lose_ratio,
                '胜率盈亏': win_rate_2,
                '盈利次数': self.win_time,
                '亏损次数': self.lose_time,
                '年化单利': (float(self.total_value - self.init_cash) / self.init_cash) * (252.0 / self.total_days) * 100,
                '权益最大回撤': self.max_drawdown_percent,
                '卡玛比率': ((float(self.total_value - self.init_cash) / self.init_cash) * (252.0 / self.total_days) * 100) / (self.max_drawdown_percent if self.max_drawdown_percent != 0 else 1),
                '最小权益': self.min_total_value,
                '最大单笔盈利': self.max_profit,
                '最大单笔亏损': self.max_loss,
                '参数组': param_info,
            }
            
            # 将新行转换为 DataFrame
            new_df = pd.DataFrame([new_row])
            
            # 检查文件是否存在
            file_exists = os.path.isfile(file_path)
            
            # 写入 CSV 文件，追加模式
            new_df.to_csv(
                file_path,
                index=False,
                mode='a',
                encoding='utf_8_sig',
                header=not file_exists
            )
            print(f"\n策略测试结果已保存至: {file_path}")
            
        # 保存交易记录
        trade_info_path = os.path.join(output_dir, 'signal_info.csv')
        self.info.to_csv(trade_info_path, index=True, mode='a', encoding='utf-8')
        print(f"\n交易记录已保存至: {trade_info_path}")
        
        # 保存每日权益记录
        equity_path = os.path.join(output_dir, self.daily_equity_file)
        self.daily_equity.to_csv(equity_path, index=False, encoding='utf-8')
        print(f"\n每日权益记录已保存至: {equity_path}")
        
        # 输出日志
        Log.log(f"最终权益:{self.total_value}", dt=current_date)
        Log.log(f"权益最大回撤:{self.max_drawdown}", dt=current_date)
        Log.log(f"总交易次数:{self.total_trade_time}", dt=current_date)
        Log.log(f"最大单笔盈利:{self.max_profit:.2f}", dt=current_date)
        Log.log(f"最大单笔亏损:{self.max_loss:.2f}", dt=current_date)
        Log.log(f"盈利次数:{self.win_time}", dt=current_date)
        Log.log(f"亏损次数:{self.lose_time}", dt=current_date)

    def _clean_memory(self, save_num=500):
        """清理内存和缓存"""
        # 记录清理前的内存使用情况
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # 转换为MB
        
        # 清理DataFrame缓存
        if hasattr(self, 'info') and len(self.info) > save_num:
            # 只保留最近的记录
            self.info = self.info.tail(save_num)
            self.info.index = range(1, len(self.info) + 1)
            self.num_of_all = len(self.info)
            
        # 清理daily_equity DataFrame
        if hasattr(self, 'daily_equity') and len(self.daily_equity) > save_num:
            self.daily_equity = self.daily_equity.tail(save_num)
            
        # 强制垃圾回收
        gc.collect()
        
        # 检查内存使用情况
        memory_after = process.memory_info().rss / 1024 / 1024  # 转换为MB
        Log.log(f"清理缓存后内存使用: {round(memory_after, 2)}MB", dt=self.datetime.datetime(0))

    def get_margin_percent(self, data):
        """
        获取给定期货品种的保证金比例和合约乘数。
        :param data: 包含期货品种wh_code的对象
        :return: 包含保证金比例和合约乘数的字典
        """
        if not hasattr(self, '_cache') or self._cache is None:
            engine = get_engine()
            query = "SELECT wh_code, 期货名, 保证金比例, 合约乘数 FROM future_codes"
            self._cache = pd.read_sql(query, con=engine).set_index('wh_code')
            self._cache['保证金比例'] = self._cache['保证金比例'].astype(float)
            self._cache['合约乘数'] = self._cache['合约乘数'].astype(int)
        try:
            wh_code = getattr(data, '_name', None)
            row = self._cache.loc[wh_code]
            return {'margin': row['保证金比例'], 'mult': row['合约乘数'], 'future_name': row['期货名']}
        except Exception:
            return {'margin': 0.1, 'mult': 10, 'future_name': '未知合约'}

    def close_all_position(self):
        """平掉所有持仓"""
        for data in self.datas:
            position = self.getposition(data)
            if position.size != 0:
                print(f"      Closing position: {position.size} contracts")
                # 记录开仓价和当前价，用于计算盈亏
                open_price = position.price
                current_price = data.close[0]
                profit = (current_price - open_price) * position.size * self.get_margin_percent(data)['mult'] if position.size > 0 else (open_price - current_price) * (-position.size) * self.get_margin_percent(data)['mult']
                
                # 更新盈亏统计
                if profit > 0:
                    self.win_time += 1
                    self.total_profit += profit
                    self.max_profit = max(self.max_profit, profit)
                    print(f"      Profit: {profit:.2f}")
                elif profit < 0:
                    self.lose_time += 1
                    self.total_loss += abs(profit)
                    self.max_loss = min(self.max_loss, profit)
                    print(f"      Loss: {profit:.2f}")
                
                # 平仓
                self.close(data=data)

    def notify_trade(self, trade):
        """交易结束通知"""
        if trade.isclosed:
            # 交易关闭，计算盈亏
            profit = trade.pnl
            commission = trade.commission
            net_profit = profit - commission
            
            print(f"Trade closed - P&L: {profit:.2f}, Commission: {commission:.2f}, Net: {net_profit:.2f}")
            
            # 记录交易信息
            data = trade.data
            current_date = self.datetime.date(0)
            
            # 创建交易记录
            new_record = {
                '时间': current_date,
                '合约名': data.cnname,
                '信号': 'CLOSE',
                '单价': trade.price,
                '手数': trade.size,
                '总价': trade.value,
                '手续费': trade.commission,
                '可用资金': self.broker.getcash(),
                '开仓均价': trade.price,
                '品种浮盈': 0,
                '权益': self.total_value,
                '当日收盘': data.close[0],
                '已缴纳保证金': 0,
                '平仓盈亏': net_profit
            }
            
            # 添加到交易记录
            self.info = pd.concat([self.info, pd.DataFrame([new_record])], ignore_index=True)
            self.info.index = range(1, len(self.info) + 1)

if __name__ == '__main__':
    # 创建回测引擎
    cerebro = bt.Cerebro()

    # 设置初始资金
    cerebro.broker.setcash(2000000)  # 设置初始资金为200万

    # 设置交易手续费
    cerebro.broker.setcommission(commission=0.0001)  # 设置手续费为0.01%

    # --- 使用 ai_backtest_interface.py 类似的测试数据 ---
    # 1. 定义目标预测数据 (使用2024年的日期)
    predict_prices_np = np.array([2990, 2950, 2949.8, 3000, 3000.4])
    predict_dates_str_list = ['20241216', '20241217', '20241218', '20241219','20241220']
    symbol_to_test = '1200'  # 测试用螺纹钢期货
    freq_to_test = 'day'

    # 2. 确定策略的激活日期范围
    strategy_active_start_date = datetime.datetime.strptime(predict_dates_str_list[0], '%Y%m%d').date()
    strategy_active_end_date = datetime.datetime.strptime(predict_dates_str_list[-1], '%Y%m%d').date()

    # 转换为数据库格式的日期字符串
    db_start_date = strategy_active_start_date.strftime('%Y%m%d')
    db_end_date = strategy_active_end_date.strftime('%Y%m%d')

    # 3. 准备数据库连接和表名
    engine = get_engine()
    table_name = f"{symbol_to_test}_{freq_to_test}"

    # 4. 获取策略开始日期之前的最近收盘价和日期
    first_date = predict_dates_str_list[0]  # 已经是正确的格式
    prior_data_query = f"""
    SELECT trade_date, close 
    FROM {table_name} 
    WHERE trade_date < {first_date}
    ORDER BY trade_date DESC 
    LIMIT 1
    """
    prior_data = pd.read_sql(prior_data_query, engine)
    if prior_data.empty:
        print(f"错误：在 {first_date} 之前没有找到历史数据。")
        exit(1)

    prior_date = prior_data['trade_date'].iloc[0]
    prior_close = prior_data['close'].iloc[0]

    print(f"\n获取到的上一个交易日: {prior_date}")

    # 5. 获取ATR计算所需的历史数据
    atr_data_query = f"""
    SELECT trade_date, high, low, close 
    FROM {table_name} 
    WHERE trade_date <= {prior_date}
    ORDER BY trade_date DESC 
    LIMIT 27  # ATR(26) 需要27个数据点
    """
    atr_data = pd.read_sql(atr_data_query, engine)
    if len(atr_data) < 27:
        print(f"错误：没有足够的历史数据计算ATR(26)。需要27个数据点，但只找到了 {len(atr_data)} 个。")
        exit(1)

    # 6. 获取实际交易日期的数据
    trading_dates_query = f"""
    SELECT DISTINCT trade_date 
    FROM {table_name}
    WHERE trade_date >= {prior_date}
    AND trade_date <= {db_end_date}
    ORDER BY trade_date
    """
    trading_dates = pd.read_sql(trading_dates_query, engine)

    if trading_dates.empty:
        print(f"错误：在指定日期范围内没有找到交易日数据。")
        print(f"查询区间: {prior_date} 到 {db_end_date}")
        exit(1)

    # 7. 获取完整的回测数据
    earliest_date = atr_data['trade_date'].iloc[-1]
    backtest_query = f"""
    SELECT trade_date, open, high, low, close, vol
    FROM {table_name} 
    WHERE trade_date >= {earliest_date}
    AND trade_date <= {db_end_date}
    ORDER BY trade_date
    """
    df = pd.read_sql(backtest_query, engine)

    # 数据检查和转换
    if df.empty:
        print(f"错误：无法获取回测所需的数据。")
        print(f"查询区间: {earliest_date} 到 {db_end_date}")
        exit(1)

    # 转换日期格式为datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')

    print("\n数据加载情况:")
    print(f"ATR计算的历史数据: {len(atr_data)} 个交易日")
    print(f"数据日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
    print(f"总数据条数: {len(df)}")
    print(f"\n策略信息:")
    print(f"策略活动期间: {strategy_active_start_date} 到 {strategy_active_end_date}")
    print(f"预测价格序列: {predict_prices_np}")
    print(f"上一个交易日收盘价: {prior_close}")

    # 设置数据索引
    df.set_index('trade_date', inplace=True)

    # 创建DataFeed
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 使用索引作为日期
        open='open',
        high='high',
        low='low',
        close='close',
        volume='vol',
        openinterest=-1
    )

    # 8. 添加数据到回测引擎
    cerebro.adddata(data_feed)

    # 9. 添加策略
    cerebro.addstrategy(AI_KDJ_Strategy,
                       backtest_start_date=strategy_active_start_date,
                       backtest_end_date=strategy_active_end_date,
                       atr_threshold=0.005,
                       predict_prices=predict_prices_np,
                       predict_dates=predict_dates_str_list)

    # 运行回测
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('最终资金: %.2f' % cerebro.broker.getvalue())

    # 绘制结果
    cerebro.plot(style='candlestick')
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os
import psutil
import gc
from tools.db_mysql import get_engine
from tools import Log
from strategies.AI_KDJ_Strategy import AI_KDJ_Strategy
from models.lstm_predictor import PricePredictor

class BacktestEngine:
    def __init__(self, init_cash=2000000, commission=0.0001, predictor=None):
        """
        初始化回测引擎
        :param init_cash: 初始资金
        :param commission: 手续费率
        :param predictor: 预测器实例 (必须有一个predict方法)
        """
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(init_cash)
        self.cerebro.broker.setcommission(commission=commission)
        self.engine = get_engine()
        self._cache = None
        self.predictor = predictor
        
        if self.predictor is None:
            Log.log("警告：未指定预测器，回测将不会包含AI预测部分。", dt=datetime.now())
        
    def get_margin_percent(self, data):
        """
        获取给定期货品种的保证金比例和合约乘数
        :param data: 包含期货品种wh_code的对象
        :return: 包含保证金比例和合约乘数的字典
        """
        if not hasattr(self, '_cache') or self._cache is None:
            query = "SELECT wh_code, 期货名, 保证金比例, 合约乘数 FROM future_codes"
            self._cache = pd.read_sql(query, con=self.engine).set_index('wh_code')
            self._cache['保证金比例'] = self._cache['保证金比例'].astype(float)
            self._cache['合约乘数'] = self._cache['合约乘数'].astype(int)
        try:
            wh_code = getattr(data, '_name', None)
            row = self._cache.loc[wh_code]
            return {'margin': row['保证金比例'], 'mult': row['合约乘数'], 'future_name': row['期货名']}
        except Exception:
            return {'margin': 0.1, 'mult': 10, 'future_name': '未知合约'}
            
    def _clean_memory(self, save_num=500):
        """
        清理内存和缓存
        :param save_num: 保留的记录数量
        """
        # 记录清理前的内存使用情况
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # 转换为MB
        
        # 清理DataFrame缓存
        if hasattr(self, 'info') and len(self.info) > save_num:
            self.info = self.info.tail(save_num)
            self.info.index = range(1, len(self.info) + 1)
            
        # 清理daily_equity DataFrame
        if hasattr(self, 'daily_equity') and len(self.daily_equity) > save_num:
            self.daily_equity = self.daily_equity.tail(save_num)
            
        # 强制垃圾回收
        gc.collect()
        
        # 检查内存使用情况
        memory_after = process.memory_info().rss / 1024 / 1024
        Log.log(f"清理缓存后内存使用: {round(memory_after, 2)}MB", dt=datetime.now())

    def prepare_data(self, symbol, freq, start_date, end_date):
        """
        准备回测数据
        :param symbol: 期货代码
        :param freq: 频率（day/min）
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 数据DataFrame
        """
        table_name = f"{symbol}_{freq}"
        
        # 获取ATR计算所需的历史数据
        atr_data_query = f"""
        SELECT trade_date, high, low, close 
        FROM {table_name} 
        WHERE trade_date <= {start_date}
        ORDER BY trade_date DESC 
        LIMIT 27
        """
        atr_data = pd.read_sql(atr_data_query, self.engine)
        
        if len(atr_data) < 27:
            raise ValueError(f"没有足够的历史数据计算ATR(26)。需要27个数据点，但只找到了 {len(atr_data)} 个。")
            
        # 获取完整的回测数据
        earliest_date = atr_data['trade_date'].iloc[-1]
        backtest_query = f"""
        SELECT trade_date, open, high, low, close, vol
        FROM {table_name} 
        WHERE trade_date >= {earliest_date}
        AND trade_date <= {end_date}
        ORDER BY trade_date
        """
        df = pd.read_sql(backtest_query, self.engine)
        
        if df.empty:
            raise ValueError(f"无法获取回测所需的数据。查询区间: {earliest_date} 到 {end_date}")
            
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        df.set_index('trade_date', inplace=True)
        
        return df
        
    def run_backtest(self, symbol, freq, strategy_start_date, strategy_end_date, atr_threshold=0.005):
        """
        运行回测
        :param symbol: 期货代码
        :param freq: 频率（day/min）
        :param strategy_start_date: 策略开始日期
        :param strategy_end_date: 策略结束日期
        :param atr_threshold: ATR阈值
        :return: 回测结果字典
        """
        if self.predictor is None:
            raise ValueError("未指定预测器，无法运行包含预测的策略。")

        try:
            # 准备历史数据
            df = self.prepare_data(symbol, freq, strategy_start_date, strategy_end_date)
            
            # 使用传入的预测器进行预测
            predict_prices, predict_dates = self.predictor.predict(df)
            
            # 创建DataFeed
            data_feed = bt.feeds.PandasData(
                dataname=df,
                datetime=None,
                open='open',
                high='high',
                low='low',
                close='close',
                volume='vol',
                openinterest=-1
            )
            
            # 添加数据
            self.cerebro.adddata(data_feed)
            
            # 添加策略
            self.cerebro.addstrategy(
                AI_KDJ_Strategy,
                backtest_start_date=datetime.strptime(strategy_start_date, '%Y%m%d').date(),
                backtest_end_date=datetime.strptime(strategy_end_date, '%Y%m%d').date(),
                atr_threshold=atr_threshold,
                predict_prices=np.array(predict_prices),
                predict_dates=predict_dates
            )
            
            # 运行回测
            initial_value = self.cerebro.broker.getvalue()
            Log.log(f'初始资金: {initial_value:.2f}')
            
            results = self.cerebro.run()
            final_value = self.cerebro.broker.getvalue()
            Log.log(f'最终资金: {final_value:.2f}')
            
            # 获取策略实例
            strategy = results[0]
            
            # 整理回测结果
            backtest_results = {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': (final_value - initial_value) / initial_value * 100,
                'max_drawdown': strategy.max_drawdown,
                'max_drawdown_percent': strategy.max_drawdown_percent,
                'total_trades': strategy.total_trade_time,
                'win_rate': (strategy.win_time / strategy.total_trade_time * 100) if strategy.total_trade_time > 0 else 0,
                'profit_factor': abs(strategy.total_profit / strategy.total_loss) if strategy.total_loss != 0 else float('inf'),
                'signal_table': strategy.generate_signal_table(),
                'daily_equity': strategy.daily_equity,
                'trade_info': strategy.info,
                'predictions': {
                    'prices': predict_prices.tolist(),
                    'dates': predict_dates
                }
            }
            
            return backtest_results
            
        except Exception as e:
            Log.log(f"回测过程中出现错误: {str(e)}")
            raise
            
    def plot_results(self, style='candlestick'):
        """
        绘制回测结果
        :param style: 图表样式
        """
        self.cerebro.plot(style=style)
        
    def save_results(self, results, output_dir='backtest_results'):
        """
        保存回测结果
        :param results: 回测结果字典
        :param output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存预测结果
        predictions_df = pd.DataFrame({
            'date': results['predictions']['dates'],
            'predicted_price': results['predictions']['prices']
        })
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        Log.log(f"\n预测结果已保存至: {predictions_path}")
            
        # 保存信号表
        if results['signal_table'] is not None:
            signal_table_path = os.path.join(output_dir, 'AI_KDJ_signal_table.csv')
            results['signal_table'].to_csv(signal_table_path, index=False, encoding='utf-8-sig')
            Log.log(f"\n信号表已保存至: {signal_table_path}")
            
        # 保存每日权益
        equity_path = os.path.join(output_dir, 'AI_KDJ_equity.csv')
        results['daily_equity'].to_csv(equity_path, index=False, encoding='utf-8')
        Log.log(f"\n每日权益记录已保存至: {equity_path}")
        
        # 保存交易记录
        if 'trade_info' in results and results['trade_info'] is not None:
            trade_info_path = os.path.join(output_dir, 'signal_info.csv')
            results['trade_info'].to_csv(trade_info_path, index=True, mode='a', encoding='utf-8')
            Log.log(f"\n交易记录已保存至: {trade_info_path}")
        
        # 保存回测摘要
        summary = {
            '初始资金': results['initial_value'],
            '最终资金': results['final_value'],
            '总收益率': f"{results['total_return']:.2f}%",
            '最大回撤': results['max_drawdown'],
            '最大回撤百分比': f"{results['max_drawdown_percent']:.2f}%",
            '总交易次数': results['total_trades'],
            '胜率': f"{results['win_rate']:.2f}%",
            '盈亏比': f"{results['profit_factor']:.2f}"
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, 'backtest_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        Log.log(f"\n回测摘要已保存至: {summary_path}")

# 使用示例
if __name__ == '__main__':
    # 创建一个预测器实例（这里使用LSTM作为示例）
    lstm_predictor = PricePredictor()

    # 创建回测引擎实例，传入预测器
    engine = BacktestEngine(init_cash=2000000, commission=0.0001, predictor=lstm_predictor)
    
    # 设置回测参数
    symbol = '1200'  # 螺纹钢期货
    freq = 'day'
    strategy_start_date = '20241201' 
    strategy_end_date = '20241220'
    
    try:
        # 运行回测（现在包含了预测过程）
        results = engine.run_backtest(
            symbol=symbol,
            freq=freq,
            strategy_start_date=strategy_start_date,
            strategy_end_date=strategy_end_date
        )
        
        # 保存结果
        engine.save_results(results)
        
        # 绘制结果
        engine.plot_results()
        
    except Exception as e:
        Log.log(f"回测失败: {str(e)}") 
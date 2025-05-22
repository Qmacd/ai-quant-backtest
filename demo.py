from backtest_engine import BacktestEngine
from models.lstm_predictor import PricePredictor
from tools import Log

# 使用示例
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
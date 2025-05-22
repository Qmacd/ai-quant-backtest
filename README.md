# AI量化回测系统

这是一个基于Python的AI量化交易回测系统，集成了机器学习预测和传统技术分析策略，用于期货市场的回测分析。

## 主要功能

- 支持期货市场的回测分析
- 集成AI预测模型和技术分析指标
- 自动计算保证金和合约乘数
- 提供详细的回测报告和可视化分析
- 支持日线和分钟线级别的回测
- 内存优化和自动清理机制

## 系统要求

- Python 3.7+
- MySQL数据库
- 必要的Python包：
  - backtrader
  - pandas
  - numpy
  - psutil
  - mysql-connector-python

## 安装说明

1. 克隆项目到本地：
```bash
git clone [项目地址]
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 配置数据库：
- 确保MySQL服务已启动
- 在`tools/db_mysql.py`中配置数据库连接信息

## 使用方法

1. 基本回测示例：
```python
from backtest_engine import BacktestEngine

# 创建回测引擎实例
engine = BacktestEngine(init_cash=2000000, commission=0.0001)

# 设置回测参数
symbol = '1200'  # 期货代码
freq = 'day'     # 频率
predict_prices = [2990, 2950, 2949.8, 3000, 3000.4]  # 预测价格
predict_dates = ['20241216', '20241217', '20241218', '20241219', '20241220']  # 预测日期
strategy_start_date = '20241216'
strategy_end_date = '20241220'

# 运行回测
results = engine.run_backtest(
    symbol=symbol,
    freq=freq,
    predict_prices=predict_prices,
    predict_dates=predict_dates,
    strategy_start_date=strategy_start_date,
    strategy_end_date=strategy_end_date
)

# 保存结果
engine.save_results(results)

# 绘制结果
engine.plot_results()
```

## 回测结果说明

系统会生成以下回测结果文件：

- `AI_KDJ_signal_table.csv`: 交易信号表
- `AI_KDJ_equity.csv`: 每日权益记录
- `signal_info.csv`: 详细交易记录
- `backtest_summary.csv`: 回测摘要（包含收益率、最大回撤等指标）

## 注意事项

1. 确保数据库中有足够的期货历史数据
2. 回测前请检查预测数据的准确性
3. 注意内存使用，系统会自动清理缓存
4. 建议先用小规模数据测试系统功能

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[待定] 
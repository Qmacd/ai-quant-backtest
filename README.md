# AI量化回测系统

这是一个基于Python的AI量化回测系统，集成了预测模型和回测引擎，用于期货交易策略的回测和评估。系统设计为可插拔的预测模型架构，您可以轻松地接入不同的AI或其他预测方法。

## 功能特点

- **可插拔的预测模型接口**：支持接入各种预测模型（如LSTM、机器学习模型、统计模型等）
- 支持期货合约的回测
- 自动计算保证金和手续费
- 提供详细的回测报告和可视化
- 支持自定义策略参数
- 自动保存回测结果和预测数据

## 系统要求

- Python 3.8+
- backtrader
- pandas
- numpy
- scikit-learn (如果使用基于sklearn的预测模型)
- torch (如果使用基于PyTorch的预测模型，如示例中的LSTM)
- psutil

## 安装步骤

1. 克隆项目：
```bash
git clone [项目地址]
cd ai-quant-backtest
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
ai-quant-backtest/
├── backtest_engine.py      # 回测引擎主程序
├── models/
│   └── lstm_predictor.py   # LSTM预测模型示例
├── strategies/
│   └── AI_KDJ_Strategy.py  # 交易策略实现
├── tools/
│   ├── db_mysql.py        # 数据库连接工具
│   └── Log.py            # 日志工具
└── requirements.txt       # 项目依赖
```

## 使用方法

1. 创建预测器：
   - 您的预测模型需要封装成一个类，并实现一个 `predict(self, dataframe)` 方法。
   - `predict` 方法应接受一个pandas DataFrame作为输入，并返回两个列表/numpy数组：预测价格序列和对应的日期序列。
   - 示例 (`models/lstm_predictor.py`) 中展示了一个基于LSTM的预测器实现。

   ```python
   # 示例：一个简单的随机预测器
   import pandas as pd
   import numpy as np
   from datetime import timedelta

   class RandomPredictor:
       def predict(self, df, predict_days=5):
           last_price = df['close'].iloc[-1]
           predict_prices = [last_price + np.random.randn() for _ in range(predict_days)]
           last_date = df.index[-1]
           predict_dates = [(last_date + timedelta(days=i+1)).strftime('%Y%m%d') 
                            for i in range(predict_days)]
           return predict_prices, predict_dates
   ```

2. 运行回测：
```python
from backtest_engine import BacktestEngine
from models.lstm_predictor import PricePredictor # 导入您的预测器类
# from your_models import YourOtherPredictor # 如果使用其他预测器

# 实例化您的预测器
my_predictor = PricePredictor() # 或者 YourOtherPredictor()

# 创建回测引擎实例，传入预测器
engine = BacktestEngine(init_cash=2000000, commission=0.0001, predictor=my_predictor)

# 设置回测参数
symbol = '1200'  # 期货代码
freq = 'day'     # 频率
strategy_start_date = '20241216'
strategy_end_date = '20241220'

# 运行回测
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
```

3. 回测结果包含：
   - 初始资金和最终资金
   - 总收益率
   - 最大回撤
   - 交易次数和胜率
   - 预测价格数据
   - 交易信号表
   - 每日权益曲线

## 注意事项

1. 数据库配置：
   - 确保MySQL数据库已正确配置
   - 检查数据库连接参数

2. 数据要求：
   - 回测数据需要至少27天的历史数据（用于计算ATR）
   - 数据格式要求：包含trade_date, open, high, low, close, vol字段

3. 内存管理：
   - 系统会自动清理内存缓存
   - 长时间回测时注意监控内存使用

## 常见问题

1. 数据不足错误：
   - 确保有足够的历史数据用于预测
   - 检查数据表名是否正确

2. 内存溢出：
   - 适当调整_clean_memory中的save_num参数
   - 考虑分批处理大量数据

## 后续开发计划

1. 完善预测模型接口定义和文档
2. 添加更多技术指标
3. 添加实时交易接口
4. 优化回测性能

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[添加许可证信息] 
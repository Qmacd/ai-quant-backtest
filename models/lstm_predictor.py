import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PricePredictor:
    def __init__(self, model_path=None):
        self.model = LSTMPredictor()
        self.scaler = MinMaxScaler()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def prepare_data(self, df, sequence_length=10):
        """准备LSTM输入数据"""
        features = ['open', 'high', 'low', 'close', 'vol']
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X = []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
        return np.array(X)
        
    def predict(self, df, predict_days=None):
        """
        预测未来价格
        :param df: 历史数据DataFrame
        :param predict_days: 预测天数，如果为None则预测整个回测区间
        :return: 预测价格和日期
        """
        sequence_length = 10
        X = self.prepare_data(df, sequence_length)
        
        if len(X) == 0:
            raise ValueError("没有足够的数据进行预测")
            
        X = torch.FloatTensor(X)
        with torch.no_grad():
            predictions = self.model(X)
            
        # 创建一个与原始数据相同形状的数组，只填充预测的收盘价
        dummy_array = np.zeros((len(predictions), 5))
        dummy_array[:, 3] = predictions.numpy().flatten()  # 将预测值放在收盘价位置
        
        # 反归一化预测结果
        predictions = self.scaler.inverse_transform(dummy_array)[:, 3]  # 只取收盘价列
        
        # 生成预测日期
        last_date = df.index[-1]
        if predict_days is None:
            # 计算回测区间的天数
            start_date = df.index[0]
            predict_days = (last_date - start_date).days + 1
            
        # 确保预测天数和预测结果长度一致
        predict_days = min(predict_days, len(predictions))
        predictions = predictions[:predict_days]
        
        predict_dates = [(last_date + timedelta(days=i+1)).strftime('%Y%m%d') 
                        for i in range(predict_days)]
        
        # 打印调试信息
        print(f"预测天数: {predict_days}")
        print(f"预测价格数量: {len(predictions)}")
        print(f"预测日期数量: {len(predict_dates)}")
        
        return predictions, predict_dates 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def analyze_profits(csv_path='signal_info.csv'):
    """分析交易盈亏情况"""
    # 设置中文字体
    try:
        plt.rcParams['font.family'] = ['Microsoft YaHei']
    except:
        try:
            plt.rcParams['font.family'] = ['Source Han Sans CN']
        except:
            try:
                plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']
            except:
                print("警告：未能找到中文字体，图表中的中文可能无法正确显示")

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gb2312')

    # 1. 计算总盈亏
    总盈亏 = df.iloc[-1]['权益'] - df.iloc[0]['权益']
    print(f"\n总盈亏: {总盈亏:,.2f}")

    # 2. 计算分品种盈亏
    平仓记录 = df[df['平仓盈亏'].notna()]
    分品种盈亏 = 平仓记录.groupby('合约名')['平仓盈亏'].sum()

    print("\n分品种盈亏:")
    for 品种, 盈亏 in sorted(分品种盈亏.items(), key=lambda x: x[1], reverse=True):
        if 品种 != 'ALL':
            print(f"{品种}: {盈亏:,.2f}")

    # 3. 绘制品种盈亏变化图
    plot_profit_changes(df)

def plot_profit_changes(df):
    """绘制品种盈亏变化图"""
    品种盈亏字典 = {}
    所有品种 = df['合约名'].unique()
    所有品种 = [品种 for 品种 in 所有品种 if 品种 != 'ALL' and isinstance(品种, str)]

    for 品种 in 所有品种:
        品种盈亏字典[品种] = []

    时间轴 = []
    当前盈亏 = {品种: 0 for 品种 in 所有品种}

    for index, row in df.iterrows():
        if isinstance(row['合约名'], str) and row['合约名'] != 'ALL':
            if pd.notna(row['平仓盈亏']):
                当前盈亏[row['合约名']] += row['平仓盈亏']
            
            for 品种 in 所有品种:
                品种盈亏字典[品种].append(当前盈亏[品种])
            
            时间轴.append(row['时间'])

    plt.figure(figsize=(20, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(所有品种)))

    for 品种, color in zip(所有品种, colors):
        plt.plot(时间轴, 品种盈亏字典[品种], label=f"{品种} ({当前盈亏[品种]:,.0f})", color=color, linewidth=1)

    plt.title('所有品种累计盈亏变化', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('累计盈亏', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('品种盈亏变化图.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("\n图表已保存为'品种盈亏变化图.png'")

if __name__ == '__main__':
    analyze_profits()
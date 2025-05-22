#峰谷赋权的一种改版：
#加入大盘，根据大盘的分数来调整做多做空的品种个数
#大盘0分 4多4空
#大盘越高，多头占比越多，反之空头占比越多


import os

import backtrader as bt
from backtrader.indicators import *
import backtrader as bt
from backtrader.indicators import *
from tools import Log 
import pandas as pd
import Indicators
import datetime
from tools.db_mysql import get_engine
import psutil
import gc
from tools import DataGet

class Top_Bottom_Double_Wenhua_Opt(bt.Strategy):
    params=(
        ('backtest_start_date', None),
        ('backtest_end_date', None),
        ('N1',10),
        ('M1',15),
        ('N2',50),
        ('M2',150),
        ('Alpha',0.7),
        ('A',25),
        ('B',25),
    )

    def __init__(self):
        self.connection=get_engine()
        self._cache = None  # 初始化缓存为None
        if not self.datas:
            raise ValueError("没有数据被加载到策略中")
        
        self.SHORTNDayHigh=dict()
        self.SHORTNDayLow=dict()
        self.LONGNDayHigh=dict()
        self.LONGNDayLow=dict()
        columns3 = ['时间', '合约名', '信号', '单价', '手数', '总价', '手续费', '可用资金','开仓均价','品种浮盈','权益','当日收盘','已缴纳保证金','平仓盈亏']
        self.NDayHigh=dict()
        self.NDayLow=dict()
        self.num_of_all=0
        self.info=pd.DataFrame(columns=columns3)#记录总的订单信息，信号明细
        self.info.index=range(1,len(self.info)+1)
        columns=['时间', '合约名','区间','分数','data名']
        self.report=pd.DataFrame(columns=columns)#记录报告
        self.report.index=range(1,len(self.report)+1)
        self.all=0
        columns2=['年化单利','胜率','盈亏比','胜率盈亏','总交易次数','盈利次数','亏损次数','参数组']
        self.final_report=pd.DataFrame(columns=columns2)#记录报告
        self.final_report.index=range(1,len(self.final_report)+1)
        self.final_report=pd.DataFrame(columns=columns2)
        self.log=dict()#在同一笔交易上记录信息
        self.init_cash=2000000#初始资金
        self.cash=2000000
        self.notify_flag=1#控制订单打印的BOOL变量
        self.order_list=dict()#订单记录
        self.paper_profit=dict()
        self.profit=dict()
        self.average_open_cost=dict()
        self.margin=dict()
        self.total_value=0
        self.is_trade=dict()
        self.is_opt=1#参数优化模式开启
        self.total_trade_time=0#总交易次数
        self.win_time=0#盈利的次数
        self.lose_time=0#亏损的次数
        self.total_profit=0#总盈利
        self.total_loss=0#总亏损
        self.total_days=0#总天数
        self.max_drawdown=0#最大回撤
        self.max_drawdown_percent=0
        self.max_total_value = self.init_cash  # 初始化为初始资金
        self.min_total_value = self.init_cash  # 初始化为初始资金
        self.nums_of_open=8
        #总共交易8个品种
        for data in self.datas:
            high=data.high
            low=data.low
            data.cnname=self.get_margin_percent(data)['future_name']
            self.order_list[data]=[0]
            self.paper_profit[data]=0
            self.profit[data]=0
            self.log[data]=0
            self.average_open_cost[data]=0
            self.margin[data]=0#这个来记录现在已经缴纳的保证金
            self.is_trade[data]=False#初始化是没有交易
            self.SHORTNDayHigh[data]=Indicators.NDayHigh(high,period=self.params.N1)
            self.SHORTNDayLow[data]=Indicators.NDayLow(low,period=self.params.M1)
            self.LONGNDayHigh[data]=Indicators.NDayHigh(high,period=self.params.N2)
            self.LONGNDayLow[data]=Indicators.NDayLow(low,period=self.params.M2)
            
    def prenext(self):
        pass

    def next(self):
        current_date = self.datetime.date(0)
        test_data=self.datas[0]

        if self.params.backtest_start_date <= current_date <= self.params.backtest_end_date:
            self.total_days+=1



            if self.cal_next_bar_is_last_trading_day(test_data):
                self.close_all_position()
            
            else:
                # print("##########################################")
                for data in self.datas:
                    c=data.close[0]
                    if c==-1:
                        continue

                    weight=self.params.Alpha
                    HSHORT=self.SHORTNDayHigh[data][0]
                    LSHORT=self.SHORTNDayLow[data][0]
                    HLONG=self.LONGNDayHigh[data][0]
                    LLONG=self.LONGNDayLow[data][0]
                    HH=weight*HSHORT+(1-weight)*HLONG
                    LL=weight*LSHORT+(1-weight)*LLONG
                    width=HH-LL
                    TOP=HH-width*(self.params.A/100)
                    BOTTOM=LL+width*(self.params.B/100)
                    if c<=BOTTOM:
                        symbol="看空区间"
                    
                    elif c>BOTTOM and c<=TOP:
                        symbol="中性区间"

                    else:
                        symbol="看多区间"

                    try:
                        score1=abs(c-LSHORT)/(HSHORT-LSHORT)
                        score2=abs(c-LLONG)/(HLONG-LLONG)
                    except ZeroDivisionError:
                        score1=0
                        score2=0
                    score=self.params.Alpha*score1+(1-self.params.Alpha)*score2
                    self.report.loc[self.all]=[current_date,data.cnname,symbol,score,data]
                    self.all+=1
                    if data.cnname=="文华商品":
                        wenhua_score=(score-0.5)*200  # 将分数转换为-100到100的范围
                        
                self.report=self.report[self.report['时间']==current_date]
                self.report=self.report[self.report['合约名']!="文华商品"]
                self.report=self.report.sort_values(by=['分数'],ascending=False)
                
                # # 根据文华商品的分数决定多空配比
                # if wenhua_score >= 70:
                #     # 强多头行情：前6做多，后2做空
                #     buy_wait = self.report.head(6)['data名'].tolist()
                #     sell_wait = self.report.tail(2)['data名'].tolist()
                # elif wenhua_score <= -70:
                #     # 强空头行情：前2做多，后6做空
                #     buy_wait = self.report.head(2)['data名'].tolist()
                #     sell_wait = self.report.tail(6)['data名'].tolist()
                # else:
                #     # 震荡行情：前4做多，后4做空
                #     buy_wait = self.report.head(4)['data名'].tolist()
                #     sell_wait = self.report.tail(4)['data名'].tolist()

                # 根据文华商品的分数决定多空配比
                if wenhua_score == 0:
                    # 震荡行情：平均分配
                    long_pos = self.nums_of_open // 2
                    short_pos = self.nums_of_open - long_pos
                elif wenhua_score > 0:
                    # 多头行情：多头数量增加
                    long_pos = (self.nums_of_open // 2) + int((self.nums_of_open // 2) * (wenhua_score / 100))
                    short_pos = self.nums_of_open - long_pos
                else:
                    # 空头行情：空头数量增加
                    short_pos = (self.nums_of_open // 2) + int((self.nums_of_open // 2) * (abs(wenhua_score) / 100))
                    long_pos = self.nums_of_open - short_pos

                # 获取交易列表
                buy_wait = self.report.head(long_pos)['data名'].tolist()
                sell_wait = self.report.tail(short_pos)['data名'].tolist()  
                
                for data in buy_wait:
                    if self.getposition(data).size==0:
                        fund=self.init_cash
                        margin_mult=self.get_margin_percent(data)
                        margin_percent=margin_mult['margin']
                        mult=margin_mult['mult']
                        lots=(fund*0.1)//(data.close[0]*margin_percent*mult)
                        self.buy(data=data,size=lots)
                    elif self.getposition(data).size<0:
                        fund=self.init_cash
                        margin_mult=self.get_margin_percent(data)
                        margin_percent=margin_mult['margin']
                        mult=margin_mult['mult']
                        lots=(fund*0.1)//(data.close[0]*margin_percent*mult)
                        self.close(data=data)
                        self.buy(data=data,size=lots)
                        
                for data in sell_wait:
                    if self.getposition(data).size==0:
                        fund=self.init_cash
                        margin_mult=self.get_margin_percent(data)
                        margin_percent=margin_mult['margin']
                        mult=margin_mult['mult']
                        lots=(fund*0.1)//(data.close[0]*margin_percent*mult)
                        self.sell(data=data,size=lots)
                    elif self.getposition(data).size>0:
                        fund=self.init_cash
                        margin_mult=self.get_margin_percent(data)
                        margin_percent=margin_mult['margin']
                        mult=margin_mult['mult']
                        lots=(fund*0.1)//(data.close[0]*margin_percent*mult)
                        self.close(data=data)
                        self.sell(data=data,size=lots)    

                for data in self.datas:
                    if data.close[0]==-1:
                        continue
                    #if data!=buy_wait and data!=sell_wait and self.getposition(data).size!=0:
                    if data not in buy_wait and data not in sell_wait and self.getposition(data).size!=0:
                        self.close(data=data)


                        
                        
                for data2 in self.datas:
                    if data2.close[0]==-1:
                        continue

                    if self.is_trade[data2]==False:#如果找到了今天没有交易的品种
                        margin_mult=self.get_margin_percent(data2)
                        #margin_percent=margin_mult['margin']
                        future_name = margin_mult['future_name']
                        mult=margin_mult['mult']#提取保证金比例和合约乘数
                        position_size = self.getposition(data2).size
                        position_size_abs = abs(position_size)
                        close_value = data2.close[0]
                    #if self.is_trade[data2]==False:#如果找到了今天没有交易的品种
                        if position_size>0:#如果持多仓
                            self.paper_profit[data2]=(close_value-self.average_open_cost[data2])*position_size_abs*mult
                            #就算不交易，还是有浮盈的变化
                            #margin=abs(self.getposition(data2).size)*abs(data2.close[0])*mult*margin_percent
                            #self.margin[data2]=margin
                            #就算不交易，也需要重新计算保证金
                            total_margin=0
                            self.total_value=0
                            for data1 in self.datas:
                                total_margin+=self.margin[data1]
                                self.total_value=self.total_value+self.margin[data1]
                                self.total_value=self.total_value+self.paper_profit[data1]
                            #权益=可用资金+已缴纳的保证金+全品种浮盈
                            self.total_value+=self.cash
                            if self.is_opt!=1:#如果不是参数优化模式
                                self.info.loc[self.num_of_all]=[current_date,future_name,"不交易",0,0,0,0,round(self.cash),round(self.average_open_cost[data2]),round(self.paper_profit[data2]),round(self.total_value),data2.close[0],round(total_margin),None]
                                self.num_of_all+=1#临时充当dataframe的行数
                        #elif self.getposition(data2).size<0:#如果持空仓
                        elif position_size<0:
                            self.paper_profit[data2]=-(close_value-self.average_open_cost[data2])*position_size_abs*mult
                            #margin=abs(self.getposition(data2).size)*abs(data2.close[0])*mult*margin_percent
                            #self.margin[data2]=margin
                            total_margin=0
                            self.total_value=0
                            for data1 in self.datas:
                                total_margin+=self.margin[data1]
                                self.total_value=self.total_value+self.margin[data1]
                                self.total_value=self.total_value+self.paper_profit[data1]
                    
                            self.total_value+=self.cash
                            if self.is_opt!=1:#如果不是参数优化模式
                                self.info.loc[self.num_of_all]=[current_date,future_name,"不交易",0,0,0,0,round(self.cash),round(self.average_open_cost[data2]),round(self.paper_profit[data2]),round(self.total_value),close_value,round(total_margin),None]
                                self.num_of_all+=1#临时充当dataframe的行数
                            #注意到这其实影响了真正的交易次数
                        #就算不交易，也需要重新计算保证金

                total_margin=0
                self.total_value=0
                #重置总保证金、总权益
                
                for data1 in self.datas:
                    if data1.close[0]==-1:
                        continue

                    total_margin+=self.margin[data1]
                    self.total_value=self.total_value+self.margin[data1]
                    self.total_value=self.total_value+self.paper_profit[data1]
                    
                self.total_value+=self.cash
                self.max_total_value = max(self.total_value, self.max_total_value)
                self.min_total_value = min(self.total_value, self.min_total_value)
                self.max_drawdown_percent = max(self.max_drawdown_percent, (self.max_drawdown/self.max_total_value)*100)
                if self.total_value < self.max_total_value:
                    current_drawdown = self.max_total_value - self.total_value
                    self.max_drawdown = max(self.max_drawdown, current_drawdown)
                            # 检查爆仓条件
                if self.total_value <= 0:  # 如果权益小于等于0
                    Log.log(f"触发爆仓条件！当前权益为负数:{self.total_value:.2f}", dt=current_date)
                    self.close_all_position()  # 平掉所有仓位
                    self.env.runstop()  # 停止回测
                    return

                if self.is_opt!=1:#如果不是参数优化模式
                    self.info.loc[self.num_of_all]=[current_date,"ALL",None,0,0,0,0,round(self.cash),None,None,round(self.total_value),None,round(total_margin),None]
                    self.num_of_all+=1#临时充当dataframe的行数

                for data2 in self.datas:
                    if data2.close[0]==-1:
                        continue

                    self.paper_profit[data2]=0#一个bar过完之后要重置浮盈
                    self.is_trade[data2]=False#每天都要初始化一次，设置成每个品种都没有交易

    def notify_order(self,order):
        #在一笔订单完成后输出有关于这笔订单的信息，这部分也可参照BackTrader源文档
        if self.notify_flag:#控制订单打印的BOOL变量为真
            flag=0
            data=order.data#获取这笔订单对应的品类
            self.is_trade[data]=True#记录该品类今天有交易
            current_time=self.datetime.date(0)#时间
            #dataname=order.data._name#合约名称
            dataname=data.cnname#中文名称
            unit_price=order.executed.price#单价
            abs_unit_price=abs(unit_price)
            trade_nums=order.executed.size#手数
            abs_trade_nums=abs(trade_nums)
            trade_value=0#总价,这个要独立计算
            #trade_value=order.margin
            trade_comm=abs(order.executed.comm)
            #手续费
            trade_type=None
            self.total_value=0
            margin_mult=self.get_margin_percent(data)
            margin_percent=margin_mult['margin']
            mult=margin_mult['mult']
            future_name = margin_mult['future_name']
            total_margin=0
            close_profit=None
            position_size = self.getposition(data).size
            position_size_abs=abs(position_size)
            close_value=data.close[0]
            close_value_abs=abs(close_value)
            if order is None:
                Log.log(self,f'Receive a none order',dt=current_time)
                return
            if order.status in [order.Submitted, order.Accepted]:
                return
            
            if order.status in [order.Completed]:
                
                if order.isbuy():#如果是买单，注意买单分四种：开多/加多/平空/减空

                    # Log.log(
                    # f"订单完成:买单,{dataname}, 手数:{(trade_nums)},"
                    # f"每手价格:{unit_price:.2f},"
                    # #f"总价格:{(order.executed.value):.2f},"
                    # f"手续费:{trade_comm:.2f},"
                    # f"该品种现有持仓:{position_size}",
                    # dt=current_time
                    # )
                    self.order_list[data].append(position_size)
                    #order_list[data]记载了data这个品类每笔交易上持仓数量的变化
                    #方便我们判断开平仓
                    #注意在BackTrader系统当中，空头的持仓为负数
                    if self.order_list[data][-1]>0 and self.order_list[data][-2]>=0 and self.order_list[data][-1]>self.order_list[data][-2]:
                    #如果现在和这笔订单完成之前，持仓都为正，且现在比之前更大，那么就是开多仓/加多仓
                        #self.cashflow(data,-1,order)#调整可用现金
                        #available_cash=self.cash#调整后获取现金
                        #total_value=self.broker.get_value()
                        if self.order_list[data][-2]==0:#开多仓
                            trade_type="开多仓"
                            self.log[data]=self.log[data]+abs_unit_price*abs_trade_nums
                            #统计开仓总成本
                            self.average_open_cost[data]=abs(self.log[data]/(position_size))
                            #在该订单执行完毕后，更新此笔交易的平均开仓成本(以收盘价计)
                            #注意这里不要以保证金计，要用收盘价计
                            margin=abs_unit_price*abs_trade_nums*mult*margin_percent
                            margin=abs(margin)
                            #本次开仓的保证金
                            self.margin[data]+=margin
                            #更新保证金情况
                            self.paper_profit[data]=(close_value_abs-abs(self.average_open_cost[data]))*position_size_abs*mult
                            #开多仓浮盈:
                            #(开完的那一天的收盘价-平均开仓成本)*持仓手数
                            self.cash=self.cash-margin
                            self.cash=self.cash-trade_comm
                            #可用资金(现金)的变化:要减掉交出的保证金和手续费
                            trade_value=margin
                            #开仓总价=保证金+手续费
                            #记录浮盈


                        else:
                            trade_type="加多仓"
                            self.log[data]=self.log[data]+abs_unit_price*abs_trade_nums
                            #增加成本
                            margin=abs_unit_price*abs_trade_nums*mult*margin_percent
                            margin=abs(margin)
                            #此次操作多加的保证金
                            self.margin[data]+=margin
                            #更新保证金
                            self.average_open_cost[data]=abs(self.log[data]/(position_size))
                            #在该订单执行完毕后，更新此笔交易的平均开仓成本(以收盘价计)
                            self.cash=self.cash-margin
                            self.cash=self.cash-trade_comm
                            #可用资金(现金)的变化:要减掉交出的保证金和手续费
                            trade_value=margin
                            #开仓总价=保证金+手续费
                            self.paper_profit[data]=(close_value-self.average_open_cost[data])*position_size_abs*mult
                            #浮盈=(收盘价-开仓均价)*持仓手数
                            

                    if self.order_list[data][-1]<=0 and self.order_list[data][-2]<0 and self.order_list[data][-1]>self.order_list[data][-2]:
                    #如果现在和这笔订单完成之前，持仓都为负，且现在比之前更大，那么就是平空仓/减空仓
                        #self.cashflow(data,1,order)#调整可用现金
                        #available_cash=self.cash#调整后获取现金
                        #total_value=self.broker.get_value()
                        if self.order_list[data][-1]==0:#平空仓
                            trade_type="平空仓"
                            self.total_trade_time+=1
                            #记录浮盈
                            #margin=abs(order.executed.price)*abs(order.executed.size)*mult*margin_percent
                            margin=self.margin[data]
                            #平仓释放的保证金
                            margin=abs(margin)
                            self.margin[data]=0
                            #平仓了，该品类保证金归零
                            self.cash=self.cash+margin
                            #可用现金要加上退回来的保证金
                            self.cash=self.cash-trade_comm
                            #减掉手续费
                            if unit_price<self.average_open_cost[data]:
                            #如果现在的收盘价小于平均开仓成本，说明空头盈利
                                self.cash+=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                self.win_time+=1#盈利次数+1
                                self.total_profit+=abs(close_profit)#记录盈利
                                self.profit[data]+=abs(close_profit)
                            else:
                            #如果现在的收盘价大于等于平均开仓成本，说明空头亏损(至少是不盈利)
                                self.cash-=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=-abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                self.lose_time+=1#亏损次数+1
                                self.total_loss+=abs(close_profit)#记录亏损
                                self.profit[data]-=abs(close_profit)
                            trade_value=margin#该笔交易总额:退回来的保证金
                            flag=1#平仓之后要把总成本和平均持仓成本全部变成0
                            self.paper_profit[data]=0
                            #已平仓，浮盈归零
                        else:
                            trade_type="减空仓"
                            self.total_trade_time+=1
                            #margin=abs(order.executed.price)*abs(order.executed.size)*mult*margin_percent
                            margin=self.margin[data]*abs_trade_nums/abs(self.order_list[data][-2])
                            #减仓释放的保证金
                            self.margin[data]-=margin
                            #更新保证金
                            self.cash=self.cash+abs(margin)
                            #可用现金要加上退回来的保证金
                            self.cash=self.cash-trade_comm
                            #减去手续费
                            self.log[data]=self.log[data]-abs(order.executed.size)*self.average_open_cost[data]
                            #减仓时候要把减的那些手的开仓成本从总成本当中去掉
                            trade_value=margin
                            if unit_price<self.average_open_cost[data]:
                                #如果现在的收盘价小于平均开仓成本，说明空头盈利
                                self.cash+=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                close_profit=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                                self.win_time+=1#盈利次数+1
                                self.total_profit+=abs(close_profit)#记录盈利
                                self.profit[data]+=abs(close_profit)
                            else:
                            #如果现在的收盘价大于等于平均开仓成本，说明空头亏损(至少是不盈利)
                                self.cash-=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=-abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                self.lose_time+=1
                                self.total_loss+=abs(close_profit)#记录亏损
                                self.profit[data]-=abs(close_profit)
                            self.paper_profit[data]=-(data.close[0]-self.average_open_cost[data])*position_size_abs*mult
                            #浮盈=-(收盘价-开仓均价)*持仓手数*乘数

                    #观察日志，发现手数和金额同号的时候是开/加仓，反之是平/减仓
                    #if (order.executed.size*order.executed.value)>0:
                        #self.cashflow(data,-1,order)
                        
                    #elif (order.executed.size*order.executed.value)<0:
                        #self.cashflow(data,1,order)
                    

                
                elif order.issell():#卖单的情况，同上
                    # Log.log(
                    # f"订单完成:卖单,{dataname},手数:{trade_nums},"
                    # f"每手价格:{unit_price:.2f},"
                    # #f"总价格:{(order.executed.value):.2f},"
                    # f"手续费:{trade_comm:.2f},"
                    # f"该品种现有持仓:{position_size}",
                    # dt=current_time
                    # )
                    self.order_list[data].append(position_size)
                    if self.order_list[data][-1]>=0 and self.order_list[data][-2]>0 and self.order_list[data][-1]<self.order_list[data][-2]:
                        #self.cashflow(data,1,order)#平多仓/减多仓
                        #available_cash=self.cash#调整后获取现金
                        #total_value=self.broker.get_value()
                        if self.order_list[data][-1]==0:#平多仓
                            trade_type="平多仓"
                            self.total_trade_time+=1
                            self.paper_profit[data]=0
                            #平仓了，浮盈归零
                            #margin=abs(order.executed.price)*abs(order.executed.size)*mult*margin_percent
                            margin=self.margin[data]
                            #平仓释放保证金
                            self.cash=self.cash+abs(margin)#现金要加上退回来的保证金
                            self.cash=self.cash-trade_comm
                            self.margin[data]=0
                            #平仓了，该品类保证金归零
                            #减去手续费
                            if self.average_open_cost[data]<unit_price:#如果平均开仓成本<收盘价，说明多头盈利
                                self.cash+=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                self.win_time+=1#盈利次数+1
                                self.total_profit+=abs(close_profit)#记录盈利
                                self.profit[data]+=abs(close_profit)
                            else:#如果平均开仓成本>=收盘价，说明多头不盈利
                                self.cash-=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=-abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                self.lose_time+=1#亏损次数+1
                                self.total_loss+=abs(close_profit)#记录亏损
                                self.profit[data]-=abs(close_profit)
                            flag=1#平仓之后要把总成本和平均持仓成本全部变成0
                            trade_value=margin

                        else:
                            trade_type="减多仓"
                            self.total_trade_time+=1
                            self.log[data]=self.log[data]-abs_trade_nums*self.average_open_cost[data]
                            #注意：减多仓要把减的那些手的开仓成本去掉，不然影响后面的计算
                            margin=self.margin[data]*abs_trade_nums/self.order_list[data][-2]
                            #平仓释放的保证金
                            self.margin[data]-=margin
                            #平仓该品种保证金减少一部分
                            self.paper_profit[data]=(close_value-self.average_open_cost[data])*position_size_abs*mult
                            #浮盈=(收盘价-开仓均价)*持仓手数*乘数
                            self.cash=self.cash+abs(margin)#现金要加上退回来的保证金
                            self.cash=self.cash-trade_comm
                            #减去手续费
                            if self.average_open_cost[data]<unit_price:#如果平均开仓成本<收盘价，说明多头盈利
                                self.cash+=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                self.win_time+=1#盈利次数+1
                                self.total_profit+=abs(close_profit)#记录盈利
                                self.profit[data]+=abs(close_profit)
                            else:#如果平均开仓成本>=收盘价，说明多头不盈利
                                self.cash-=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                                #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                                close_profit=-abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                                self.lose_time+=1#亏损次数+1
                                self.total_loss+=abs(close_profit)#记录亏损
                                self.profit[data]-=abs(close_profit)
                            trade_value=margin


                    if self.order_list[data][-1]<0 and self.order_list[data][-2]<=0 and self.order_list[data][-1]<self.order_list[data][-2]:
                        #self.cashflow(data,-1,order)#开空仓/加空仓
                        #available_cash=self.cash#调整后获取现金
                        #total_value=self.broker.get_value()
                        if self.order_list[data][-2]==0:
                            trade_type="开空仓"
                            self.log[data]=self.log[data]+abs_unit_price*abs_trade_nums
                            self.average_open_cost[data]=abs(self.log[data]/(position_size))
                            self.paper_profit[data]=-(abs(close_value)-abs(self.average_open_cost[data]))*position_size_abs*mult
                            #开空仓，浮盈是0
                            margin=abs_unit_price*abs_trade_nums*mult*margin_percent
                            #开仓存入的保证金
                            self.margin[data]=margin
                            self.cash-=margin
                            self.cash-=trade_comm
                            #可用资金(现金)的变化:要减掉交出的保证金和手续费
                            #记录浮盈
                            #在该订单执行完毕后，更新此笔交易的平均开仓成本(以收盘价计)
                            #注意这里不要以保证金计，要用收盘价计
                            trade_value=margin
                        else:
                            trade_type="加空仓"
                            self.log[data]=self.log[data]+abs_unit_price*abs_trade_nums
                            #更新开仓成本
                            margin=abs_unit_price*abs_trade_nums*mult*margin_percent
                            #开仓存入的保证金
                            self.margin[data]+=margin
                            self.cash-=margin
                            self.cash-=trade_comm
                            self.average_open_cost[data]=abs(self.log[data]/(position_size))
                            self.paper_profit[data]=-(close_value-self.average_open_cost[data])*position_size_abs*mult
                            #浮盈=-(收盘价-开仓均价)*持仓手数*乘数
                            trade_value=margin
            
            if trade_type==None:#反手开仓是特殊条件,BackTrader无法判断这种条件
                if self.order_list[data][-2]<0 and position_size>0:
                    #此时是处在平空反手开多过程当中
                        self.total_trade_time+=1
                        trade_type="反手平空"
                        #记录浮盈
                        margin=self.margin[data]
                        #平仓释放的保证金
                        margin=abs(margin)
                        self.margin[data]=0
                        #平仓了，该品类保证金归零
                        self.cash=self.cash+abs(margin)
                        #可用现金要加上退回来的保证金
                        self.cash=self.cash-trade_comm
                        #减掉手续费
                        if unit_price<self.average_open_cost[data]:
                        #如果现在的收盘价小于平均开仓成本，说明空头盈利
                            self.cash+=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                            #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                            close_profit=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                            self.win_time+=1#盈利次数+1
                            self.total_profit+=abs(close_profit)#记录盈利
                            self.profit[data]+=abs(close_profit)
                        else:
                        #如果现在的收盘价大于等于平均开仓成本，说明空头亏损(至少是不盈利)
                            self.cash-=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                            #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                            close_profit=-abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                            self.lose_time+=1#亏损次数+1
                            self.total_loss+=abs(close_profit)#记录亏损
                            self.profit[data]-=abs(close_profit)
                        trade_value=margin#该笔交易总额:退回来的保证金
                        flag=1#平仓之后要把总成本和平均持仓成本全部变成0
                        self.paper_profit[data]=0
                        #已平仓，浮盈归零
                        self.order_list[data].append(0)

                if self.order_list[data][-2]>0 and position_size<0:
                    #此时是处在平多反手开空过程当中
                        self.total_trade_time+=1
                        trade_type="反手平多"
                        #记录浮盈
                        margin=self.margin[data]
                        #平仓释放的保证金
                        margin=abs(margin)
                        self.margin[data]=0
                        #平仓了，该品类保证金归零
                        self.cash=self.cash+abs(margin)
                        #可用现金要加上退回来的保证金
                        self.cash=self.cash-trade_comm
                        #减掉手续费
                        if unit_price>self.average_open_cost[data]:
                        #如果平仓价大于平均开仓成本，说明多头盈利
                            self.cash+=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                            #盈利额=手数*|(收盘价-平均开仓成本)|*乘数
                            close_profit=abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                            self.win_time+=1#盈利次数+1
                            self.total_profit+=abs(close_profit)#记录盈利
                            self.profit[data]+=abs(close_profit)
                        else:
                        #如果现在的收盘价大于等于平均开仓成本，说明空头亏损(至少是不盈利)
                            self.cash-=abs_trade_nums*abs(unit_price-self.average_open_cost[data])*mult
                            #亏损额=手数*|(收盘价-平均开仓成本)|*乘数
                            close_profit=-abs_trade_nums*abs(abs_unit_price-abs(self.average_open_cost[data]))*mult
                            self.lose_time+=1#亏损次数+1
                            self.total_profit+=abs(close_profit)#记录亏损
                            self.profit[data]-=abs(close_profit)
                        trade_value=margin#该笔交易总额:退回来的保证金
                        flag=1#平仓之后要把总成本和平均持仓成本全部变成0
                        self.paper_profit[data]=0
                        #已平仓，浮盈归零
                        self.order_list[data].append(0)



        for data1 in self.datas:
            total_margin+=self.margin[data1]
            self.total_value=self.total_value+self.margin[data1]
            self.total_value=self.total_value+self.paper_profit[data1]
                
        self.total_value+=self.cash
                #权益计算方式:现金+各品种已缴纳的保证金+各品种的浮动盈亏
                #现金计算方式:期初现金-手续费-保证金支出(若有)+保证金收入(若有)

                    #观察日志，发现手数和金额同号的时候是开/加仓，反之是平/减仓
                    #if (order.executed.size*order.executed.value)>0:
                        #self.cashflow(data,-1,order)
                        
                    #elif (order.executed.size*order.executed.value)<0:
                        #self.cashflow(data,1,order)
        if self.is_opt!=1:#如果不是参数优化模式
            self.info.loc[self.num_of_all]=[current_time,future_name,trade_type,unit_price,trade_nums,round(trade_value),round(trade_comm),round(self.cash),round(self.average_open_cost[data]),round(self.paper_profit[data]),round(self.total_value),close_value,round(total_margin),close_profit]
            self.num_of_all+=1#交易次数+1
        if flag==1:
            self.log[data]=0
            self.average_open_cost[data]=0

    def stop(self):
        current_date = self.datetime.date(0)
        self.report.to_csv('score.csv',index=True,mode='a',encoding='gbk')
        param_info=[]
        skip_attr=['notdefault','isdefault']
        for name in dir(self.params):
            if not name.startswith('_') and name not in skip_attr:
                value=getattr(self.params,name)
                param_info.append(f'{name}={value}')

        if self.total_trade_time!=0:#有交易
            win_rate=float(self.win_time)/self.total_trade_time
            #胜率
            try:
                win_lose_ratio=(float(self.total_profit)/self.win_time)/(float(self.total_loss)/self.lose_time)
            except ZeroDivisionError:
                win_lose_ratio=float('inf')
            #盈亏比例
            win_rate_2=(1+win_lose_ratio)*win_rate
            if math.isinf(win_rate_2):
                win_rate_2=float('inf')

            file_path = f'峰谷改_调大盘动态仓位.csv'

            # 创建新的数据行
            new_row = {
                '胜率': win_rate * 100,
                '总交易次数': self.total_trade_time,
                '盈亏比': win_lose_ratio,
                '胜率盈亏': win_rate_2,
                '盈利次数': self.win_time,
                '亏损次数': self.lose_time,
                '年化单利': (float(self.total_value - self.init_cash) / self.init_cash) * (
                            252.0 / self.total_days) * 100,
                '权益最大回撤':self.max_drawdown_percent,  # 改为百分比
                '卡玛比率':((float(self.total_value - self.init_cash) / self.init_cash) * (
                            252.0 / self.total_days) * 100)/(self.max_drawdown_percent),
                '参数组': param_info,
                '最小权益':self.min_total_value,
            }

            # 将新行转换为 DataFrame
            new_df = pd.DataFrame([new_row])

            # 检查文件是否存在
            file_exists = os.path.isfile(file_path)

            # 写入 CSV 文件，追加模式
            new_df.to_csv(
                file_path,
                index=False,  # 不写入行索引
                mode='a',  # 追加模式
                encoding='utf_8_sig',  # 使用带 BOM 的 UTF-8 编码
                header=not file_exists  # 如果文件不存在，则写入表头；否则不写入
            )
        if self.is_opt!=1:#如果不是参数优化模式
            self.info.to_csv('signal_info.csv',index=True,mode='a',encoding='utf-8')
        Log.log(f"最终权益:{self.total_value}",dt=current_date)
        Log.log(f"权益最大回撤:{self.max_drawdown}",dt=current_date)

        DataGet.clear_memory_cache()
        for attr in dir(self):
            if isinstance(getattr(self, attr), pd.DataFrame) or isinstance(getattr(self, attr), pd.Series):
                delattr(self, attr)
    def _load_cache(self):
        """首次调用时加载数据到缓存"""
        if self._cache is None:
            try:
                engine = get_engine()
                query = "SELECT wh_code, 期货名, 保证金比例, 合约乘数 FROM future_codes"
                self._cache = pd.read_sql(query, con=engine).set_index('wh_code')
            finally:
                engine.dispose()  # 确保连接被关闭


    def get_margin_percent(self, data):
        """
        获取给定期货品种的保证金比例和合约乘数。

        :param data: 包含期货品种wh_code的对象
        :return: 包含保证金比例和合约乘数的字典
        """
        self._load_cache()  # 确保缓存已加载

        try:
            wh_code = data._name
            row = self._cache.loc[wh_code]
            return {'margin': row['保证金比例'], 'mult': row['合约乘数'], 'future_name':row['期货名']}
        except KeyError:
            raise ValueError(f"未找到 {wh_code} 的保证金比例或合约乘数")

    def cal_next_bar_is_last_trading_day(self,data):
        try:
            next_next_day=data.datetime.date(1)
            if next_next_day>=self.params.backtest_end_date:
                return True
            else:
                return False
        except IndexError or ValueError:
            return False


    def close_all_position(self):#全平
        for data in self.datas:
            if data.close[0]==-1:
                continue

            self.close(data=data)
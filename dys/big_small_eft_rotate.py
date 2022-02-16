# -*- coding: UTF-8 -*-

import traceback
from kupy.logger import logger
from kupy.config import configs
from kupy.dbadaptor import DBAdaptor
from dys.domain import AccountData, BigSmallEtfRotate, StrategyAccount
from dys.stockutil import StockUtil

import os.path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# pd.set_option("expand_frame_repr", False)

df_cache_file = configs["data_folder"].data + os.path.basename(__file__).replace(".py", "") + "_cache.pkl"

class BigSmallEtfRotateStrategy:
    def __init__(self, account_id:int):
        #检查初始数据是否正确
        self.db:DBAdaptor = DBAdaptor()
        self.util:StockUtil = StockUtil()
        account:StrategyAccount = self.db.get_any_by_id(StrategyAccount, 1) 

        if account is None:
            logger.error(f"invest.strategy_acount 账户{account_id} 初始状态为空，或者不正确，请立即修复!")
            raise Exception(f"没有找到账户{account_id} 初始状态为空，或者不正确，请立即修复!")
        #
        if account.amount < 1000: 
            raise Exception("初始金额不能小于 1000")

        self.account = account

        self.ticker_x = configs["ticker_x"].data
        self.ticker_y = configs["ticker_y"].data
        self.trade_rate = float(configs["trade_rate"].data)/10000
        self.strategy_status_cd = configs["strategy_status_cd"].data
        if not self.is_valid_status_cd(status=self.strategy_status_cd):
            raise Exception(f"stategy_status_cd 设置不正确, {self.strategy_status_cd}")
        self.momentum_days=int(configs["momentum_days"].data)
        df_ticker = self.db.get_df_by_sql(f"select distinct ticker, sec_short_name from stock.fund where ticker='{self.ticker_x}' or ticker = '{self.ticker_y}'")
        if df_ticker.shape[0] != 2:
            raise Exception(f"查询ticker 基金名字出错! 查询结果为: {df_ticker}")
        self.ticker_x_name = df_ticker.loc[df_ticker["ticker"]==self.ticker_x, "sec_short_name"].iloc[0]
        self.ticker_y_name = df_ticker.loc[df_ticker["ticker"]==self.ticker_y, "sec_short_name"].iloc[0]
    
    @staticmethod
    def is_valid_status_cd(status:str)->bool:
        """判断策略的部署状态是否合法
        R-un 正常运行
        S-uspend 长时间刮起策略,如果已有买入的股票，则卖出股票
        P-ause 临时暂停策略运行，跳过当日策略行为，什么都不操作

        Args:
            status (str): _description_

        Returns:
            bool: _description_
        """        
        return status in ['R', 'S', 'P']

    def set_strategy_satus(self, status:str):
        if not self.is_valid_status_cd(status):
            raise Exception(f"strategy_status_cd值不合法 code: {status}!")
        self.strategy_status_cd = status

    def deploy(self, start_date:datetime):
        """根据给定的日期进行部署策略，部署策略之后，结果会写入数据库
        同一个账户，同一天的数据不能覆盖，不能重复,否则会抛出异常
        结果会把给定日期到当前日期的策略结果写入数据库。

        Args:
            start_date (datetime): 部署的起始日期
        """        
        if not self.util.is_trade_day(start_date=start_date):
            logger.info(f"{start_date} 不是交易日，跳过计算")
            pass
        df = self.get_strategy_by_date(start_date)
        self.write_df_to_db(df)
        pass

    def get_strategy_by_date(self, start_date:datetime, end_date:datetime=None)->pd.DataFrame:
        """根据给定日期获取start_date之后的策略数据 包含start_date

        Args:
            start_date (datetime): 启动日期为可交易etf基金的有效日期, 如果给定日期不可交易则抛出异常
        
        Returns:
            Dataframe:
                trade_date
                ticker_x
                open_price_x
                close_price_x
                chg_pct_x
                ticker_y
                open_price_y
                close_price_y
                chg_pct_y
                big_mom ->大盘动量值
                small_mom ->小盘动量值
                style -> 行情风格倾向
                pos -> 当日持股情况 big,small,empty
                trade_cd -> 交易指令 0 不交易 1 买入,2卖出,3换仓  交易指令来自于比较当日持仓风格比较前日持仓风格
                next_trade_cd -> 下一个交易日交易指令 
                trade_time -> 交易日期
                strategy_chg_pct -> 轮动收益率 
                strategy_chg_pct_adj -> 调整手续费后轮动收益
                big_net -> 大盘行情 净值走势参考
                small_net -> 小盘行情 净值走势参考
                strategy_net -> 轮动行情 净值参考
        """
        if not self.util.is_trade_day(start_date): 
            logger.error(f"输入的日期:{start_date} 不是可交易日期!")
            raise Exception(f"输入的日期:{start_date} 必须是可交易日期!")

        mom_begin_date = self.util.get_trade_day_by_offset(start_date, self.momentum_days + 1)
        df_load = self.load_fundd_data(mom_begin_date, end_date=end_date)
        df_load = df_load.sort_values(
                    by=["ticker", "trade_date"]
                ).drop_duplicates()

        df_load = df_load[
            df_load["trade_date"]
            >= mom_begin_date
        ]

        df_big = df_load[df_load["ticker"] == self.ticker_x.replace("'", "")]
        df_small = df_load[df_load["ticker"] == self.ticker_y.replace("'", "")]

        df = pd.merge(df_big, df_small, on=["trade_date"], how="inner")

        df["big_mom"] = df["close_price_x"].pct_change(periods=self.momentum_days)
        df["small_mom"] = df["close_price_y"].pct_change(periods=self.momentum_days)

        df.loc[df["big_mom"] > df["small_mom"], "style"] = "big"
        df.loc[df["big_mom"] < df["small_mom"], "style"] = "small"
        df.loc[(df["big_mom"] < 0) & (df["small_mom"] < 0), "style"] = "empty"
        df["style"].fillna(method="ffill", inplace=True)
        #判断今日持仓, 今日持仓即昨日的风格值
        df["pos"] = df["style"].shift(1)
        df.dropna(subset=["pos"], inplace=True)
        df.reset_index(inplace=True)

        #第一天必须是空仓，因为是盘后部署，只有盘后部署才能获取当天数据
        df.loc[0, "pos"]= "empty"

        df["pre_pos"] = df["pos"].shift(1)
        # df.loc[0, "pre_pos"] = "init"
        df.loc[0,"trade_cd"] = 0

        #判断今日操作指令
        df.loc[( df["pos"] == df["pre_pos"] ), "trade_cd"] =  0

        df.loc[( (df["pre_pos"]!=df["pos"]) & (df["pos"]!="empty") ), "trade_cd"] =  3

        df.loc[((df["pre_pos"]=="empty")) & (df["pos"]!="empty"), "trade_cd"] =  1

        df.loc[( (df["pre_pos"]=="big")|(df["pre_pos"]=="small")) & (df["pos"]=="empty"), "trade_cd"] =  2 

        df.loc[( df["pos"] != df["pos"].shift(1) ) & ( df["pos"].shift(1)==df["pos"].shift(1) ) , "trade_time"] = df["trade_date"]

        df["next_trade_cd"] = df["trade_cd"].shift(-1)
        df["next_trade_cd"].fillna(0, inplace=True)

        # df["trade_time"].fillna("", inplace=True)
        
        #初始化轮动收益
        df.loc[df["pos"] == "big", "strategy_chg_pct"] = df["chg_pct_x"]
        df.loc[df["pos"] == "small", "strategy_chg_pct"] = df["chg_pct_y"]
        df.loc[df["pos"] == "empty", "strategy_chg_pct"] = 0

        df["strategy_chg_pct_adj"] = df["strategy_chg_pct"]
        # 买入收益调整
        df.loc[(df["trade_cd"] == 1) & (df["pos"] == "big"), "strategy_chg_pct_adj"] =  (df["close_price_x"]/df["open_price_x"]) -(1+self.trade_rate)
        df.loc[(df["trade_cd"] == 1) & (df["pos"] == "small"), "strategy_chg_pct_adj"] =  (df["close_price_y"]/df["open_price_y"]) -(1+self.trade_rate)

        # 卖出收益调整
        df.loc[ (df["trade_cd"] == 2) & (df["pre_pos"] == "big"), "strategy_chg_pct_adj" ] = ( df["open_price_x"]*(1-self.trade_rate)/df["pre_close_price_x"] ) -1
        df.loc[ (df["trade_cd"] == 2) & (df["pre_pos"] == "small"), "strategy_chg_pct_adj" ] = ( df["open_price_y"]*(1-self.trade_rate)/df["pre_close_price_y"] ) -1

        # 换仓收益调整, 减两次手续费
        df.loc[(df["trade_cd"] == 3) & (df["pos"]=="big") , "strategy_chg_pct_adj"] = (df["open_price_y"] * (1 - self.trade_rate)/df["pre_close_price_y"]) -1 + (df["close_price_x"]/df["open_price_x"])-(1+self.trade_rate)
        df.loc[(df["trade_cd"] == 3) & (df["pos"]=="small") , "strategy_chg_pct_adj"] = (df["open_price_x"] * (1 - self.trade_rate)/df["pre_close_price_x"]) -1 + (df["close_price_y"]/df["open_price_y"])-(1+self.trade_rate)
        # df.set_index('trade_date')

        df["big_net"] = (1 + df["chg_pct_x"]).cumprod()
        df["small_net"] = (1 + df["chg_pct_y"]).cumprod()
        df["strategy_net"] = (1 + df["strategy_chg_pct_adj"]).cumprod()

        df.set_index("trade_date", inplace=True)
        # df.to_csv("df_" + start_date.replace("'", "") + ".csv")
        logger.debug(f"计算出来的结果:{df}")
        return df
    
    def calc_df_with_account_data(self, init_account_data:AccountData, df:pd.DataFrame)->pd.DataFrame:
        """第一行数据为初始化数据

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: 返回带有账号资金的 轮动收益数据
        """
        pre = init_account_data

        df["vol"]=0
        df["trade_fee"]=0
        df["amount"]=0
        df["total_amount"]=0
        df["balance"]=0
        df["strategy_chg_pct_real"]=0
        df["strategy_net_real"]=0

        for index, row in df.iterrows():
            try:
                cur = self.calc_new_account_data(pre, row)
            except Exception as e:
                debug_file="/tmp/calc_df_with_account_data.csv"
                df.to_csv(debug_file)
                raise Exception(f"df 状态不对, 状态保存至:{debug_file}")
            df.loc[index, "vol"]=cur.vol
            df.loc[index, "trade_fee"]=cur.trade_fee
            df.loc[index, "amount"]=cur.amount
            df.loc[index, "total_amount"]=cur.total_amount
            df.loc[index, "balance"]=cur.balance
            df.loc[index, "strategy_chg_pct_real"]=cur.chg_pct
            df.loc[index, "strategy_net_real"]=cur.net
            pre=cur



        return df
    def write_df_to_db(self, df:pd.DataFrame)->int:
        """将策略运行数据写入数据库并且计算实际的盈亏数据
           只会讲数据库中缺失的数据写入数据库，以前写入的数据库不会更新

        Args:
            df (pd.DataFrame): 根据策略跑出来的数据

        Raises:
            Exception: 如果状态不对，抛出异常

        Returns:
            int: 返回写入数据库的条目数量
        """        

        if self.strategy_status_cd == "S":
            logger.info(f"{self.account.account_id} suspended，skip writing data to db!")
            return 0
            
        account_id = self.account.account_id
        df_db=self.db.get_df_by_sql(f"select * from invest.big_small_etf_rotate where account_id = {account_id}")

        # 过滤掉db中已经有的数据
        pre = AccountData()

        if df_db is None :
            raise Exception("读取数据库失败，请检查！")
        elif df_db.shape[0]== 0:
            #初始化数据
            #初始资金
            pre.total_amount = self.account.amount
            pre.balance = pre.total_amount
            pre.net = 1
        elif df_db.shape[0] > 0:
            df = df[df.index>=df_db.iloc[-1]["trade_date"]]
            pre.vol = df_db.iloc[-1]["vol"]
            pre.amount = df_db.iloc[-1]["amount"]
            pre.total_amount = df_db.iloc[-1]["total_amount"]
            pre.balance = pre.total_amount - pre.amount
            pre.trade_fee = df_db.iloc[-1]["trade_fee"]
            pre.chg_pct = df_db.iloc[-1]["strategy_chg_pct_real"]
            pre.net = df_db.iloc[-1]["strategy_net_real"]
        
        df = self.calc_df_with_account_data(pre, df)
        count = 0 


        for index, row in df.iterrows():
            bser:BigSmallEtfRotate = BigSmallEtfRotate()
            bser.trade_date = index
            bser.account_id = account_id
            bser.account_name = self.account.account_name
            bser.strategy_status_cd = self.strategy_status_cd

            bser.ticker_x = row["ticker_x"]
            bser.sec_short_name_x = self.ticker_x_name
            bser.pre_close_price_x = row["pre_close_price_x"]
            bser.open_price_x = row["open_price_x"]
            bser.close_price_x = row["close_price_x"]
            bser.chg_pct_x = row["chg_pct_x"]
            bser.ticker_y = row["ticker_x"]
            bser.sec_short_name_y = self.ticker_y_name
            bser.pre_close_price_y = row["pre_close_price_y"]
            bser.open_price_y = row["open_price_y"]
            bser.close_price_y = row["close_price_y"]
            bser.chg_pct_y = row["chg_pct_y"]
            bser.strategy_chg_pct = row["strategy_chg_pct_adj"] # 扣除手续费后的涨幅
            bser.strategy_net = row["strategy_net"] # 净值因子

            bser.big_mom = row["big_mom"]
            bser.small_mom = row["small_mom"]
            bser.style = row["style"]
            bser.pos = row["pos"]

            bser.big_net = row["big_net"] # 净值因子
            bser.small_net = row["small_net"] # 净值因子

            if self.strategy_status_cd == "P":
                bser.trade_cd = 0 
                bser.trade_fee = 0 
                bser.vol=pre.vol
                bser.amount=pre.amount
                bser.total_amount=pre.total_amount
                bser.strategy_chg_pct_real = 0 
                bser.strategy_net_real=pre.strategy_net_real
            else:
                bser.trade_cd = row["trade_cd"] # 交易类型 0 不交易 1 买入,2卖出,3换仓
                bser.next_trade_cd = row["next_trade_cd"] # 交易类型 0 不交易 1 买入,2卖出,3换仓
                bser.vol = row["vol"] # 交易多少手
                bser.trade_fee = row["trade_fee"] # 扣除手续费后的涨幅
                bser.amount = row["amount"] # 股票市值
                bser.total_amount = row["total_amount"] # 资产市值
                bser.strategy_chg_pct_real = row["strategy_chg_pct_real"] # 实际交易中扣除手续费后的涨幅
                bser.strategy_net_real = row["strategy_net_real"] # 实际交易中净值因子

            bser.update_time = datetime.now()

            ret = self.db.save(bser)
            if not ret:
                raise Exception("插入数据库失败")
            count = count + 1

        return count

    
    def calc_new_account_data(self, pre:AccountData,  row:pd.Series)->AccountData:
        """计算交易多少股票,以及手续费
        Args:
            pre: 前一日的股票账户信息
            row (pd.Series): 轮动策略返回的行数据
            trade_date,ticker_x,open_price_x,close_price_x,chg_pct_x,ticker_y,open_price_y,close_price_y,chg_pct_y,big_mom,small_mom,style,pos,strategy_chg_pct,trade_time,strategy_chg_pct_adj,big_net,small_net,strategy_net

        Returns:
            当日交易后的AccountData
        """        
        cur = AccountData()
        if row["trade_cd"] == 0:
            cur.trade_fee=0
            change = pre.amount* row["strategy_chg_pct"]
            cur.amount=pre.amount + change
            cur.total_amount=pre.total_amount + change
            cur.balance=pre.balance
            cur.vol=pre.vol
            #vol,balance 和pre一样
            cur.chg_pct = change/pre.total_amount
            cur.net = pre.net*(1+cur.chg_pct)
            return cur
        # 买入
        elif row["trade_cd"] == 1:
            if row["pos"] == "big":
                to_buy_price=row["open_price_x"]
            elif row["pos"] == "small":
                to_buy_price=row["open_price_y"]
            
            cur.vol =  int(pre.balance/(to_buy_price*(1+self.trade_rate)*100)) * 100
            cur.amount = to_buy_price * cur.vol
            cur.total_amount=pre.total_amount - cur.amount*self.trade_rate
            cur.balance=cur.total_amount - cur.amount
            cur.trade_fee=cur.amount * self.trade_rate

        elif row["trade_cd"] == 2 :
            pre_pos_style=row["pre_pos"] 
            if pre_pos_style == "big":
                to_sale_price=row["open_price_x"]
            elif pre_pos_style == "small" :
                to_sale_price=row["open_price_y"]
            else:
                raise Exception( f"前日持仓数据有问题: {pre_pos_style} pre:{pre}, cur:{cur}" )

            cur.balance = pre.vol * to_sale_price * (1 - self.trade_rate) + pre.balance
            cur.trade_fee = pre.vol *  to_sale_price * self.trade_rate
            cur.total_amount = cur.balance
            cur.amount = 0
            cur.vol = 0
        # 换仓
        elif row["trade_cd"] == 3:
            #小盘换大盘
            if row["pos"] == "big":
                to_sale_price = row["open_price_y"]
                to_buy_price = row["open_price_x"]
            #大盘换小盘
            elif row["pos"] == "small":
                to_sale_price = row["open_price_x"]
                to_buy_price = row["open_price_y"]
            #先卖后买
            #卖出操作之后
            cur.balance = pre.balance + pre.vol * to_sale_price*(1-self.trade_rate)
            sale_trade_fee = pre.vol * to_sale_price*self.trade_rate
            #然后买入, 以开盘价买入是存在误差的，这里忽略这种误差
            cur.vol = int(cur.balance/(to_buy_price*( 1 + self.trade_rate)*100)) * 100
            cur.amount = cur.vol*to_buy_price 
            cur.balance = cur.balance - cur.amount - sale_trade_fee
            buy_trade_fee = cur.amount*self.trade_rate
            cur.trade_fee = sale_trade_fee + buy_trade_fee
            cur.total_amount=cur.amount  + cur.balance

        cur.chg_pct = (cur.total_amount - pre.total_amount )/pre.total_amount
        cur.net = pre.net * (1+cur.chg_pct)
        self.internal_account_amount_check(cur)
        return cur
    
    def internal_account_amount_check(self, cur):
        if cur.total_amount <cur.amount:
            raise Exception(f"{self.account.account_id} 账户资金状态不对")

        if cur.total_amount <cur.balance:
            raise Exception(f"{self.account.account_id} 账户资金状态不对")

        if cur.total_amount <= 0:
            raise Exception(f"{self.account.account_id} 账户资金状态不对")
        
        if cur.total_amount - cur.amount - cur.balance > 1/10000:
            raise Exception(f"{self.account.account_id} 资金状态不对: {cur}")
            
        
    def load_fundd_data(self, begin_date:datetime, end_date:datetime=None)->pd.DataFrame:
        """给定日期加载基金数据数据

        Args:
            begin_date (datetime): 其实日期，包含起始日期

        Raises:
            Exception: 如果没有数据，抛出异常

        Returns:
            pd.DataFrame: 返回df
        """        
        if end_date is not None:
            end_date_condition = f"and trade_date<='{end_date}'"
        else:
            end_date_condition = ""

        sql_command1 = (
            "select trade_date, ticker,pre_close_price, open_price,close_price, chg_pct from stock.fund_day where "
            + f"(ticker = '{self.ticker_x}' or ticker = '{self.ticker_y}') "
            + f"and trade_date >= '{begin_date}' {end_date_condition}" 
        )
        logger.debug(sql_command1)
        df = self.db.get_df_by_sql(sql_command1)
        if df.shape[0] == 0:
            logger.error(" there is no data in !")
            raise Exception("没有加载到数据!")

        return df

        
    def backtest_by_date(self, start_date:datetime, show_plot:bool)->pd.DataFrame:
        """根据给定的日期进行回测

        Args:
            start_date (datetime): 部署的起始日期
            show_plot (bool): 是否展示图表

        Returns:
            pd.DataFrame: 返回该日期的策略结果df
        """        

        # cur = start_date
        # while cur <= datetime.today().date():
        #     if not self.util.is_trade_day(cur):
        #         continue
        #     df = self.get_strategy_by_date(cur)
        #     cur = cur + timedelta(days=1)
        pass
    
    def backtest_by_range(self, date_list:list, show_plot: bool) -> pd.DataFrame:
        pass

    def run_with_plot(self, start_date:datetime):
        _backtest_sec = ["'20200716'", "'20210125'", "'20210315'", "'20210722'"]
        # 策略参数
        # 回测起始日期
        # _backtest_sec=["'20050101'" ,"'20090101'" , "'20140101'", "'20160101'","'20190101'"]
        # _backtest_sec=["'20160101'","'20190101'"]
        # _start_date="'20201223'"

        # start_time = time.time()

        for _start_date in _backtest_sec:
            # 因为计算N日动量，前N日是没有数据的，因此要提前N+8日
            #  timedelta +8 意思是隔了4个周末共8天 不考虑法定节假日
            df = self.run_as_daemon(_start_date)

            # 开启新窗口
            plt.figure()
            plt.plot(df["strategy_net"], label=_start_date + "strategy", linewidth=4)
            # 下面两条作为bench mark
            plt.plot(df["big_net"], label="big")
            plt.plot(df["small_net"], label="small")
            plt.legend()

        plt.show()

    # 计算年化收益率函数
    def annual_return(date_line, capital_line):
        """
        :param date_line: 日期序列
        :param capital_line: 账户价值序列
        :return 输出在回测期间的年化收益率
        """
        # df = pd.D
        pass
    
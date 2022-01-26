# -*- coding: UTF-8 -*-

from xxlimited import Null
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import psycopg2
import time, os.path
from datetime import datetime, timedelta
import sys
import logging

pd.set_option('expand_frame_repr', False)
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s- %(message)s')

#导入数据
# postgres config
postgres_host = "pg-quant-invest"               # 数据库地址
postgres_port = "5432"       # 数据库端口
postgres_user = "user"              # 数据库用户名
postgres_password = "password"      # 数据库密码
postgres_datebase = "market"      # 数据库名字

df_cache_file = os.path.basename(__file__).replace(".py", "") + "_cache.pkl"

# 计算年化收益率函数
def annual_return(date_line, capital_line):
    """
    :param date_line: 日期序列
    :param capital_line: 账户价值序列
    :return 输出在回测期间的年化收益率
    """
    df = pd.D

#沪深300 510300 2012-05-28
#易方达创业板ETF 159915 20111209
_ticker_x = "'510300'" # X为大市值基金
# _ticker_x = "'510050'" # X为大市值基金
_ticker_y = "'159915'" # Y为小市值基金
_trade_rate = 0.6/10000 #万分之0.6
# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
conn = psycopg2.connect(conn_string)

# 策略参数
## 回测起始日期
# _backtest_sec=["'20050101'" ,"'20090101'" , "'20140101'", "'20160101'","'20190101'"]
_backtest_sec=["'20200716'","'20210125'", "'20210315'", "'20210722'"]
# _backtest_sec=["'20160101'","'20190101'"]
# _start_date="'20201223'"
momentum_days=25

if len(sys.argv)>=2 and (sys.argv[1] == "-r" or sys.argv[1] == "--reset"):
    if os.path.exists(df_cache_file):
        os.remove(df_cache_file)
    
start_time = time.time()

if os.path.isfile(df_cache_file):
    df_load = pd.read_pickle(df_cache_file) 
    logging.info (" Load df from file, take time: " + str(time.time()-start_time))
else:
    try:
        sql_command1 = "select trade_date, ticker,open_price,close_price, chg_pct from stock.fund_day where " + \
            f"(ticker = {_ticker_x} or ticker = {_ticker_y}) " + \
            "and trade_date > to_date( '20120528' , 'yyyymmdd') " 
        logging.info(sql_command1)
        
        df_load = pd.read_sql(sql_command1, conn)

    except:
        logging.error( " load data from postgres failure !")
        exit()

    if df_load.shape[0] == 0:
        logging.warn(" there is no data in !")
    else:
        ts = time.time()
        logging.info(" Load data from DB, takes time: " + str(ts-start_time))
        #默认排序
        df_load=df_load.sort_values(by=['ticker', 'trade_date']).drop_duplicates()
        df_load.to_pickle(df_cache_file)
        logging.info(" Cache df to file, takes time:-" + str(time.time()-ts))
for _start_date in _backtest_sec:    
    # 因为计算N日动量，前N日是没有数据的，因此要提前N+8日
    #  timedelta +8 意思是隔了4个周末共8天 不考虑法定节假日
    df_load = df_load [df_load['trade_date']>pd.to_datetime(_start_date) - timedelta(days=momentum_days+11)]
    df_big = df_load[df_load['ticker'] == _ticker_x.replace("'", '') ]
    df_small= df_load[df_load['ticker'] == _ticker_y.replace("'", '') ]

    df = pd.merge(df_big,df_small, on=['trade_date'], how="inner")

    df['big_mom'] = df['close_price_x'].pct_change(periods=momentum_days)
    df['small_mom'] = df['close_price_y'].pct_change(periods=momentum_days)

    df.loc[df['big_mom'] > df['small_mom'], 'style'] = 'big'
    df.loc[df['big_mom'] < df['small_mom'], 'style'] = 'small'
    df.loc[(df['big_mom'] < 0) & (df['small_mom'] < 0), 'style'] = 'empty'
    df['style'].fillna( method = 'ffill', inplace=True)
    df['pos'] = df['style'].shift(1)
    df.dropna(subset=['pos'], inplace=True)
    logging.info("设置的_start_date:" + _start_date + " 实际的_start_date:" + str(df['trade_date'].iloc[0]))
    # Reset start_date to be actual start day
    _start_date = str(df['trade_date'].iloc[0])
    df.loc[df['pos'] == 'big', 'strategy_chg_pct'] = df['chg_pct_x']
    df.loc[df['pos'] == 'small', 'strategy_chg_pct'] = df['chg_pct_y']
    df.loc[df['pos'] == 'empty', 'strategy_chg_pct'] = 0
    df.loc[df['pos'] != df['pos'].shift(1), 'trade_time'] = df['trade_date']
    df['strategy_chg_pct_adj']=df['strategy_chg_pct']
    #买入收益调整
    df.loc[(pd.notnull(df['trade_time'])) & (df['pos'] =='big'), 'strategy_chg_pct_adj'] = df['close_price_x']/(df['open_price_x'] * ( 1+ _trade_rate)) -1
    df.loc[(pd.notnull(df['trade_time'])) & (df['pos'] =='small'), 'strategy_chg_pct_adj'] = df['close_price_y']/(df['open_price_y'] * ( 1+ _trade_rate)) -1

    #卖出收益调整
    df.loc[(pd.notnull(df['trade_time'].shift(-1))) , 'strategy_chg_pct_adj'] = (1 + df['strategy_chg_pct'])*(1-_trade_rate) -1
    # df.set_index('trade_date')
    df['big_net']=(1+df['chg_pct_x']).cumprod()
    df['small_net']=(1+df['chg_pct_y']).cumprod()
    df['strategy_net'] = (1 + df['strategy_chg_pct_adj']).cumprod()
    
    df.set_index('trade_date', inplace=True)
    df.to_csv("df_" +_start_date.replace("'", "") + ".csv")

    #开启新窗口
    plt.figure()
    plt.plot(df['strategy_net'], label=_start_date + "strategy", linewidth=4)
    #下面两条作为bench mark    
    plt.plot(df['big_net'], label="big")
    plt.plot(df['small_net'], label="small")
    plt.legend()

plt.show()

    

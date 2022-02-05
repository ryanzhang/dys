# -*- coding: UTF-8 -*-

import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import psycopg2
import time, os.path
from datetime import datetime
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

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
conn = psycopg2.connect(conn_string)

# 策略参数
## 回测起始日期
# _backtest_sec=["'20050101'" ,"'20090101'" , "'20140101'", "'20160101'","'20190101'"]
_backtest_sec=["'20140101'", "'20160101'","'20190101'"]
# _backtest_sec=["'20160101'","'20190101'"]
# _start_date="'20190101'"
_select_equ_number=10

if len(sys.argv)>=2 and (sys.argv[1] == "-r" or sys.argv[1] == "--reset"):
    if os.path.exists(df_cache_file):
        os.remove(df_cache_file)
    
output = pd.DataFrame()
for _start_date in _backtest_sec:    
    #"select * from stock.trade_calendar where is_month_end=True"
    start_time = time.time()
    sql_command2 = f"select calendar_date as trade_date from stock.trade_calendar where is_month_end=True and calendar_date > to_date( {_start_date} , 'yyyymmdd')"
    df_monthend= pd.read_sql(sql_command2, conn)
    df_monthend['trade_month']=df_monthend['trade_date'].map(lambda x: 100*x.year + x.month)

    if os.path.isfile(df_cache_file):
        df = pd.read_pickle(df_cache_file) 
        logging.info (_start_date + " Load df from file, take time: " + str(time.time()-start_time))
    else:
        try:
            sql_command1 = "select ticker,trade_date,is_open,market_value, chg_pct from stock.mkt_equ_day where " + \
                f"trade_date > to_date( {_start_date} , 'yyyymmdd') " 
            logging.info(sql_command1)
            
            df = pd.read_sql(sql_command1, conn)
        except:
            logging.error(_start_date + " load data from postgres failure !")
            exit()

        if df.shape[0] == 0:
            logging.warn(_start_date + " there is no data in !")
        else:
            ts = time.time()
            logging.info(_start_date + " Load data from DB, takes time: " + str(ts-start_time))

            df.to_pickle(df_cache_file)
            logging.info(_start_date + " Cache df to file, takes time:-" + str(time.time()-ts))
    df=df[df['trade_date']>pd.to_datetime(_start_date).date()]
    # print(df.info())
    df['trade_month']=df['trade_date'].map(lambda x: 100*x.year + x.month)
    df['abs_chg_pct']=df['chg_pct']+1
    df['cum_chg_pct']=df.groupby(["ticker", "trade_month"])['abs_chg_pct'].cumprod()
    df['open_days']=df.groupby(["ticker", "trade_month"])['is_open'].cumsum()

    # df.to_csv("df_d.cvs")
    #求每个月最后一个交易日的成交情况

    df_monthly = pd.merge(df, df_monthend, on=['trade_date', 'trade_month'], how='inner')
    df_monthly=df_monthly.sort_values(by=['ticker', 'trade_date']).drop_duplicates()
    
    df_monthly['cum_chg_pct']=df_monthly['cum_chg_pct']-1
    # quit()
    # print(df_monthly.head(10))
    # quit()
    #求每个月月涨幅并且含有当月交易天数
    # df_monthly_sum = df.groupby(["ticker", "trade_month"]).sum()
    # df_monthly['open_days']=(df.groupby(["ticker", "trade_month"])['is_open']).sum()
    df_monthly.to_csv("df_monthly.csv")

    # # df_monthly_sum.to_csv("df_monthly_sum.csv")
    # df_monthly=pd.merge(df_monthly, df_monthly_sum, on=['ticker', 'trade_month'], how='inner')
    # df_monthly.to_csv("df_monthly1.csv")
    df_monthly['chg_pct_next_month']=df_monthly['cum_chg_pct'].shift(-1).ffill()
    # df_monthly=df_monthly.drop('market_value_y', 1)
    # df_monthly.to_csv("df_monthly2.csv")
    # print(df_monthly.head(10))

    # 过滤掉一些不适宜交易的股票
    df_monthly = df_monthly [df_monthly['is_open'] !=0 ]
    df_monthly = df_monthly [ df_monthly['open_days']>=10 ]
    df_monthly = df_monthly [ df_monthly['chg_pct']<=0.097]

    output[_start_date + '_所有股票下月涨幅'] = df_monthly.groupby('trade_date')['chg_pct_next_month'].mean()
    df_monthly['market_value_rank'] = df_monthly.groupby('trade_date')['market_value'].rank()
    # df_monthly.to_csv("mkt_rank.csv")
    df_monthly=df_monthly[df_monthly['market_value_rank']<=_select_equ_number]
    output[_start_date +'_选中股票下月涨幅'] = df_monthly.groupby('trade_date')['chg_pct_next_month'].mean()

    df_monthly['ticker'] += ' '
    output[_start_date + '_股票代码'] = df_monthly.groupby('trade_date')['ticker'].sum()
    output[_start_date+ '_line_benchmark'] = (output[_start_date + '_所有股票下月涨幅']+1).cumprod()
    output[_start_date +'_line'] = (output[_start_date + '_选中股票下月涨幅']+1).cumprod()
    plt.figure()
    plt.plot(output[_start_date+ '_line'], label=_start_date, linewidth=2)
    plt.plot(output[_start_date+'_line_benchmark'], label=_start_date + '_benchmark') #黄色
    plt.legend()


logging.info(" Total execution takes time: " + str(time.time()-start_time))
output.to_csv("output.csv")
# plt.legend(loc='best')
# plt.legend(handles=[one, two, three], title="title",
#                     loc=4, fontsize='small', fancybox=True)
plt.show()
quit()

# print(df_monthly_sum.unstack(level=0).head())
# df['trade_month']=pd.to_datetime(df['trade_date'], 'yyyy-mm')

# df_monthly = df.set_index('trade_date').merge(df_monthend.set_index('trade_date'), on=['trade_date'])
print(type(df_monthend['trade_date'][0]))
print(type(df['trade_date'][0]))

df_monthly = pd.merge(df, df_monthend, on=['trade_date'], how='inner')
print(df_monthly.head())
# df_monthly=df[df["trade_date"].isin(df_monthend["trade_date"])]
# print(df_monthly.head())
# df['monthly_trade_days'] = 
print ("Total Executed Time(s):-", time.time()-start_time)

# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import time, os.path

pd.set_option('expand_frame_repr', False)

#导入数据
# postgres config
postgres_host = "postgresql"               # 数据库地址
postgres_port = "5432"       # 数据库端口
postgres_user = "user"              # 数据库用户名
postgres_password = "password"      # 数据库密码
postgres_datebase = "market"      # 数据库名字
df_cache_file = os.path.basename(__file__) + "_cache.pkl"

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
conn = psycopg2.connect(conn_string)

start_time = time.time()
if os.path.isfile(df_cache_file):
    df = pd.read_pickle(df_cache_file) 
    print ("Load df from file, take time:-", time.time()-start_time)
else:
    try:
        sql_command1 = "select ticker,trade_date,is_open,market_value  from stock.mkt_equ_day where " + \
            "trade_date > to_date('20200101', 'yyyymmdd') " + \
            "and is_open = True " + \
            "and chg_pct <= 0.097"
        print(sql_command1)
        df = pd.read_sql(sql_command1, conn)
    except:
        print("load data from postgres failure !")
        exit()

    if df.shape[0] == 0:
        print("there is no data in !")
    else:
        ts = time.time()
        print("Load data from DB, takes time:-", ts-start_time)

        df.to_pickle(df_cache_file)
        print("Cache df to file, takes time:-", time.time()-ts)
# df.info()
df['monthly_trade_days'] = 
print ("Total Executed Time(s):-", time.time()-start_time)

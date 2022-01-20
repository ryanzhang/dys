# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import time

pd.set_option('expand_frame_repr', False)

#导入数据
# postgres config
postgres_host = "postgresql"               # 数据库地址
postgres_port = "5432"       # 数据库端口
postgres_user = "user"              # 数据库用户名
postgres_password = "password"      # 数据库密码
postgres_datebase = "market"      # 数据库名字
table_mkt_equ_d = "stock.mkt_equ_day"           #数据库中的表的名字

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
conn = psycopg2.connect(conn_string)

# sql_command1 = "select * from " + table_mkt_equ_d + " where sec_id='000001.XSHE' "
sql_command1 = "select * from " + table_mkt_equ_d + " where trade_date > to_date('20160101', 'yyyymmdd')" + \
    " and is_open = True and chg_pct <= 0.097"
print(sql_command1)
start_time = time.time()
try:
    stock_data = pd.read_sql(sql_command1, conn)
except:
    print("load data from postgres failure !")
    exit()

if stock_data.shape[0] == 0:
    print("there is no data in !")

# stock_data.columns = [i.encoding('utf8') for i in stock_data.columns ]
# stock_data['交易日期'] = pd.to_datatime([stock_data['交易日期'])
stock_data.info()
# stock_data.to_csv("/tmp/merged.csv")
print ("Executed Time(s):-", time.time()-start_time)

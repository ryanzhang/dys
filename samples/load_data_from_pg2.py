# -*- coding: UTF-8 -*-

import pandas as pd
# import matplotlib.pyplot as plt
import psycopg2
import time


# pd.set_option('expand_fram_repr', False)

#导入数据
# postgres config
# postgres_host = "192.168.2.20"               # prod 数据库地址
# postgres_host = "192.168.2.13"               # staging 数据库地址
postgres_host = "postgresql"               # dev 数据库地址
postgres_port = "5432"       # 数据库端口
postgres_user = "user"              # 数据库用户名
postgres_password = "password"      # 数据库密码
postgres_datebase = "market"      # 数据库名字
postgres_table1 = "stock.mkt_equ_day"           #数据库中的表的名字

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
start_time = time.time()

conn = psycopg2.connect(conn_string)

sql_command1 = "select * from " + postgres_table1
print (sql_command1)

try:
    data1 = pd.read_sql(sql_command1, conn)
except:
    print("load data from postgres failure !")
    exit()

if data1.shape[0] == 0:
    print("there is no data in !")

print(type (data1['trade_date'][0]))

print ("Executed Time(s):-", time.time()-start_time)


# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# pd.set_option('expand_fram_repr', False)

#导入数据
# postgres config
postgres_host = "192.168.2.13"               # 数据库地址
postgres_port = "5432"       # 数据库端口
postgres_user = "user"              # 数据库用户名
postgres_password = "password"      # 数据库密码
postgres_datebase = "market"      # 数据库名字
postgres_table1 = "equity"           #数据库中的表的名字

# connection string
conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_datebase + \
                " user=" + postgres_user + " password=" + postgres_password
conn = psycopg2.connect(conn_string)

sql_command1 = "select * from" .format(postgres_table1)

sql_command = "select * from stock.equity where sec_id='000001.XSHE'"

curs = conn.cursor()
curs.execute(sql_command)
data = pd.DataFrame(curs.fetchall())
print (data)
# try:
#     data1 = pd.read_sql(sql_command1, conn)
# except:
#     print("load data from postgres failure !")
#     exit()

# if data1.shape[0] == 0:
#     print("there is no data in !")

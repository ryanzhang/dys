# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import time

pd.set_option('expand_frame_repr', False)


start_time = time.time()
try:
    stock_data = pd.read_csv('/tmp/merged.csv', encoding='utf8')
except:
    print("load data from csv failure !")
    # exit()

if stock_data.shape[0] == 0:
    print("there is no data in !")

# stock_data.columns = [i.encoding('utf8') for i in stock_data.columns ]
# stock_data['交易日期'] = pd.to_datatime([stock_data['交易日期'])
print (type (stock_data['trade_date'][0]))
# stock_data.to_csv("/tmp/merged.csv")
print ("Executed Time(s):-", time.time()-start_time)

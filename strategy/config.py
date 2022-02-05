# -*- coding: UTF-8 -*-
import os

from jproperties import Properties

configs = Properties()
default_setting = {
    "postgres_host": "pg-quant-invest",
    "postgres_port": "5432",
    "postgres_user": "",
    "postgres_password": "",
    "postgres_database": "market",
    "log_level": "INFO",
    "log_output_path": "",
    "data_folder": "/tmp/strategy/cache",
}
# 设置默认值
for (key, value) in default_setting.items():
    configs.setdefault(key, value)

with open("strategy/resources/app-config.properties", "rb") as config_file:
    configs.load(config_file)
    if configs.get("cache_folder") is not None:
        cache_folder = configs["cache_folder"].data
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, 755)

    if not configs["log_output_path"]:
        log_file_path = configs["log_output_path"].data
        log_filename = os.path.basename(log_file_path)
        log_folder = log_file_path.replace(log_filename, "")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder, 755)

# syd -同步我的金融数据

[![codecov](https://codecov.io/gh/ryanzhang/syd/branch/main/graph/badge.svg?token=syd_token_here)](https://codecov.io/gh/ryanzhang/syd)
[![CI](https://github.com/ryanzhang/syd/actions/workflows/main.yml/badge.svg)](https://github.com/ryanzhang/syd/actions/workflows/main.yml)

<!-- 找到缺失的数据, 并更新数据，数据来源是通达信，使用 mootdx -->

设计思路

* 查询上一次最新更新日期，每一个表对应一个日期， 然后获取到最新日期的数据 并且更新数据库.
* 第一次运行会从第一天开始查询，设置的其实日期为 1991-07-03
* 数据库大部分数据都是实现导入进去的，所以本代码主要是负责添加增量数据
* 本代码部署的时候，按照每日16:30进行启动，因为这个时间点可以获取到最新一天的数据
* 数据库表接口 由另一个微服务kyd进行维护


## Install it from PyPI

```bash
pip install syd
```

## Usage

```py
from syd import BaseClass
from syd import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m syd
#or
$ syd
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

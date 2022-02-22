from datetime import date, datetime, timedelta
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.bs import BaseStrategy
import pandas as pd

from dys.domain import TradeModel


class MyStrategy(BaseStrategy):
    def __init__(self):
        BaseStrategy.__init__(self)
        logger.debug("Construct MyStrategy")
        pass

    def set_mkt_timing_alg(self) -> bool:
        return True

    def set_equ_pool(self) -> bool:
        db = DBAdaptor(is_use_cache=True)

        df = db.get_df_by_sql(
            "select * from stock.equity where exchange_cd = 'XSHG'"
        )

        self.df_equ_pool = df

        logger.debug(f"自定义股票池已经设定成功")
        return True

    def set_equd_pool(self) -> bool:
        db = DBAdaptor(is_use_cache=True)

        df = db.get_df_by_sql(
            "select * from stock.mkt_equ_day where trade_date >= '20220104'"
        )

        self.df_equd_pool = df

        logger.debug(f"自定义股票池已经设定成功")
        return True

    def set_select_equ_condition(self, query_str) -> bool:
        """设置选股条件字符串，条件字符串按照df.query接受的语法

        Args:
            query_str (_type_): _description_

        Returns:
            bool: 是否成功
        """
        self.select_equ_condition = query_str
        logger.debug(f"选股条件已经设置完毕{query_str}")
        return True
    
    def post_hook_select_equ(self):
        self.df_choice_equd.dropna(inplace=True)
        logger.debug(f"post hook after 选股 has been triggered {self.df_choice_equd.shape[0]}!")
        pass

    def set_rank_factors(self, rankfactors: list) -> bool:
        """设置排序因子列表

        Args:
            rankfactors (list): _description_

        Returns:
            bool: _description_
        """
        self.rank_factors = rankfactors
        logger.debug(f"排序因子已经设置完毕{rankfactors}")

        return True

    def set_trade_model(self) -> bool:
        """设置交易模型

        Args:
            rankfactors (list): _description_

        Returns:
            bool: _description_
        """
        self.trade_model = TradeModel()
        logger.debug(f"交易模型已经设定{self.trade_model}")
        return True


class MyETFStrategy(BaseStrategy):
    def __init__(self):
        BaseStrategy.__init__(self, trade_type=1)
        logger.debug("Construct MyETFStrategy")

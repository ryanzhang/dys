from datetime import date, datetime, timedelta
from importlib import invalidate_caches
import math
import pytest
import pandas as pd
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.bs import BaseStrategy
from dys.domain import RankFactor
from dys.metric import m


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail


class TestDomains:
    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")

    def test_rank_factor_construct_ok(self):
        series = pd.Series([1, 2, 3, 4, 5])
        rf = RankFactor(m.momentum, 20, series, metric_name="20日动量")
        assert rf is not None

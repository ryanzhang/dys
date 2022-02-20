
from datetime import date, datetime, timedelta
from importlib import invalidate_caches
import math
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger
import pandas as pd


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

class TestBigSmallEtfRotateStrategy:
    @pytest.fixture(scope="class")
    def db(self):
        return DBAdaptor()
    

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")
        
    def test_strategy_construct_ok(self, db:DBAdaptor):
        pass
    
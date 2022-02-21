
from datetime import date, datetime, timedelta
from importlib import invalidate_caches
import math
import pytest
from kupy.dbadaptor import DBAdaptor
from kupy.logger import logger

from dys.base import Base


given = pytest.mark.parametrize
skipif = pytest.mark.skipif
skip = pytest.mark.skip
xfail = pytest.mark.xfail

class MyStrategy(Base):
    def __init__(self):
        Base.__init__(self)
        logger.debug("Construct MyStrategy") 
        pass
    

class TestBase:
    @pytest.fixture(scope="class")
    def db(self):
        pass
    

    @pytest.fixture(autouse=True)
    def setup_teamdown(self):
        logger.info("TestCase Level Setup is triggered!")
        yield
        logger.info("TestCase Level Tear Down is triggered!")
        
    def test_using_base_construct_ok(self, db:DBAdaptor):
        ms = MyStrategy()
        pass
    

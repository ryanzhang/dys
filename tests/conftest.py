# -*- coding: UTF-8 -*-
import sys
import pytest
from syd.logger import logger

# # 初始化日志
# logging.basicConfig(
#     level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s"
# )

# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    logger.info(
        "\n=======================request start================================="
    )
    logger.debug(
        "\n=======================request start================================="
    )
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield

# -*- coding: UTF-8 -*-
"""
dys base module.

This is the principal module of the kupy project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""


from kupy.logger import logger


class Base:
    def __init__(self):
        logger.info("Construct dys Base")
        pass

    def base_method(self) -> str:
        """
        Base method.
        """
        return "hello from dys Base Class"

    def __call__(self) -> str:
        return self.base_method()


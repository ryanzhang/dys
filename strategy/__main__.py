# -*- coding: UTF-8 -*-
import argparse  # pragma: no cover

from syd.logger import logger
from syd.stocksyncer import StockSyncer  # pragma: no cover

from . import BaseClass, base_function  # pragma: no cover


def main() -> None:  # pragma: no cover
    """
    The main function executes on commands:
    `python -m syd` and `$ syd `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    parser = argparse.ArgumentParser(
        description="syd 同步我的金融数据库",
        epilog="Enjoy the syd functionality!",
    )
    # # This is required positional argument
    # parser.add_argument(
    #     "name",
    #     type=str,
    #     help="The username",
    #     default="ryanzhang",
    # )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Optionally adds verbosity",
    )
    args = parser.parse_args()
    # print(f"{args.message} {args.name}!")
    if args.verbose:
        print("Verbose mode is on.")
        # logging.basicConfig(
        #     level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s"
        # )

    logger.info("Executing main function")
    ss = StockSyncer()
    ss.sync_trade_calendar()
    ss.sync_equity()
    ss.sync_mkt_equ_d()
    ss.sync_fund_day()
    logger.info("End of main function")


if __name__ == "__main__":  # pragma: no cover
    main()

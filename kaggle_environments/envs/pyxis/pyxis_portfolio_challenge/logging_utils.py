import logging


def setup_logging(
    level: int | str = logging.INFO,
) -> None:
    """
    Sets the root logger and it's level. To be used at entry points to code.

    To set up a root logger, use the following:
        > logging.basicConfig(level=logging.INFO)
        > logger = initialize_logger(logger_name=__name__, level=logging.INFO)
    Example format for logging:
        2021-01-26 14:38:03,734  aiml_mkrb_ltr.utils.common  [INFO]: Message...

    Parameters
    ----------
    level : [type], optional
        [description], by default logging.INFO

    """
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s  %(name)s  [%(levelname)s]: %(message)s")
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level=level)
    logger.addHandler(console_handler)

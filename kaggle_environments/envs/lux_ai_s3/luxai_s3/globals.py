import os

TERM_COLORS = True
try:
    from termcolor import colored

    TERM_COLORS = (
        os.environ["LUX_COLORS"] == "False" if "LUX_COLORS" in os.environ else True
    )
except:
    TERM_COLORS = False
    print("termcolor not installed, skipping dependency")
    pass

def print_green(s: str, end: str = '\n') -> None: print("\033[92m{}\033[00m".format(s), end=end)
def print_yellow(s: str, end: str = '\n') -> None: print("\033[93m{}\033[00m".format(s), end=end)
def print_red(s: str, end: str = '\n') -> None: print("\033[91m{}\033[00m".format(s), end=end)
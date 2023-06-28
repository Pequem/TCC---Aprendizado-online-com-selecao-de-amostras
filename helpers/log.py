import time


class Log:

    script_start = None

    def __init__(self) -> None:
        if not 'SCRIPT_TIME_START' in globals():
            self.script_start = time.time()

    def debug(self, message):
        elapsed_time = round((time.time() - self.script_start) * 1000)
        print("{:010d}".format(elapsed_time) + ': ' + message)

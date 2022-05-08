import time

class Log:

    def __init__(self) -> None:
        if not 'SCRIPT_TIME_START' in globals():
            global SCRIPT_TIME_START
            SCRIPT_TIME_START = time.time()


    def degub(self, message):
        global SCRIPT_TIME_START
        elapsed_time = round((time.time() - SCRIPT_TIME_START) * 1000)
        print("{:010d}".format(elapsed_time) +  ': ' + message)
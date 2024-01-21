from datetime import datetime


def get_time():
    return datetime.today().strftime('%m-%d-%H-%M-%S')

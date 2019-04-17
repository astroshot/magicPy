# coding=utf-8

import datetime as dt


def get_date_str():
    now = dt.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')

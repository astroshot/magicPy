# coding=utf-8


class DataError(BaseException):
    def __init__(self, message, data=None):
        self.message = message
        self.data = dict(data) if data else None

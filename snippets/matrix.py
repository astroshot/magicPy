# coding=utf-8
"""
"""


class MatrixLink(object):
    """Matrix Link

    """

    def __init__(self, dims):
        self.dims = dims
        self.num = len(dims) - 1  # num of matrix
        self.steps = [[0 for i in range(self.num)] for j in range(self.num)]

    def _adjust_order(self, i, j):
        if i == j:
            return 0

        if i + 1 == j:
            return 0

        k = 1
        while k < self.num:
            count = self._adjust_order(i, k) + self._adjust_order(k + 1, self.num) + \
                    self.dims[i] * self.dims[k] * self.dims[j]
            self.steps[k][i] = count

    def _adjust(self):
        i = 1
        j = self.num
        k = 1
        while i < self.num:
            while k < self.num:
                self.steps[i][j] = self.steps[i][k] + self.steps[k + 1][j] + \
                                   self.dims[i] * self.dims[k] * self.dims[j]
                k += 1

    def solve(self):
        pass


if __name__ == '__maim__':
    dims = (30, 35, 15, 5, 10, 20, 25)
    m = MatrixLink(dims)
    m.solve()

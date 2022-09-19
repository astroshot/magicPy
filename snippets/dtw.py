# coding=utf-8

import numpy as np


class DTW:
    """Dynamic Time Warping
    """
    def __init__(self, test_arr, target_arr):
        self.test_arr = test_arr
        self.target_arr = target_arr
        x, y = np.meshgrid(test_arr, target_arr)
        # 距离矩阵
        self.dist = np.abs(x - y)
        self.cost = None

    def solve_cost(self):
        m, n = np.shape(self.dist)
        cost = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    cost[i, j] = self.dist[i, j]
                    continue
                if i == 0:
                    cost[i, j] = cost[i, j-1] + self.dist[i, j]
                    continue
                if j == 0:
                    cost[i, j] = cost[i-1, j] + self.dist[i, j]
                    continue
                cost[i, j] = min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1]) + self.dist[i, j]

        self.cost = cost
        return cost[m-1, n-1]

    def solve_path(self):
        path = []
        m, n = np.shape(self.dist)
        val = 0
        for i in range(m, -1, -1):
            for j in range(n, -1, -1):
                if i == m-1 and j == n-1:
                    val = self.cost[i, j]
                    continue
                if i == m-1:
                    array_slice = self.cost[i-1:i, j-1:j]
                    np.where()




if __name__ == '__main__':
    target_arr = [3, 5, 6, 7, 7, 1]
    # test_arr = [3, 6, 6, 7, 8, 1, 1]
    test_arr = [2, 5, 7, 7, 7, 7, 2]
    solver = DTW(test_arr, target_arr)
    print(solver.solve_cost())


# coding=utf-8
"""
题目：给一个数组，将数组分成两个子数组，使得这两个子数组和的差值最小。
思路：0-1 背包问题的变种，设置背包大小为数组各元素和的一半，尽量去装满这个背包。
状态转移方程：solve(i, target) = max(solve(i-1, target-arr[i])+arr[i], solve(i-1, target))
solve(i, target) 表示数组 arr 中不超过 i 下标，背包大小为 target 时能装下的最大值
"""


class Bag(object):

    def __init__(self, arr):
        self.arr = arr
        self.arr_sum = sum(arr)
        self.res = {}
        self.selected = set()

    def solve(self, i, target):
        k = '{}-{:.6f}'.format(i, target)
        v = self.res.get(k)
        if v is not None:
            return v

        if i == 0:
            if self.arr[i] > target:
                return 0

            self.res[k] = self.arr[i]
            return self.arr[i]

        if target < self.arr[i]:
            return self.solve(i - 1, target)

        tmp1 = self.solve(i - 1, target - self.arr[i]) + self.arr[i]
        tmp2 = self.solve(i - 1, target)

        if tmp1 >= tmp2:
            self.res[k] = self.arr[i]
            return tmp1
        else:
            return tmp2


if __name__ == '__main__':
    arr = [2, 4, 6, 7, 8, 10]
    # arr = [2, 3, 4, 5]
    bag = Bag(arr)

    c = bag.solve(len(arr) - 1, bag.arr_sum / 2)
    print(c)
    print(bag.res)

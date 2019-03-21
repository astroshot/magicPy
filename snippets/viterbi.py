# coding=utf-8

"""
Viterbi Algorithm Example:

0 -> 1 -> 2 ->3

"""

import numpy as np


class Viterbi(object):

    def __init__(self, graph):
        shape = np.shape(graph)
        assert shape[0] == shape[1]

        self.graph = graph
        self.best_score = np.zeros(shape[0])
        self.best_edge = [None] * shape[0]
        self.best_path = []

    def __forward(self):
        nodes = range(1, len(self.best_edge))
        graph = self.graph

        for node in nodes:
            self.best_score[node] = np.inf
            edges = self.graph[:, node]
            for i in range(len(edges)):
                if edges[i] <= 0:
                    continue
                score = self.best_score[i] + edges[i]
                if score < self.best_score[node]:
                    self.best_score[node] = score
                    self.best_edge[node] = (i, node)
        print("edges: {}".format(self.best_edge))
        print("score: {}".format(self.best_score))

    def __backward(self):
        num = len(self.best_edge)
        next_edge = self.best_edge[num - 1]
        while next_edge is not None:
            self.best_path.append(next_edge)
            next_edge = self.best_edge[next_edge[0]]
        self.best_path.reverse()
        print("best path: {}".format(self.best_path))

    def solve(self):
        self.__forward()
        self.__backward()


def make_graph():
    graph = np.zeros((4, 4))
    graph[0][1] = 2.5
    graph[0][2] = 1.4
    graph[1][2] = 4.0
    graph[1][3] = 2.1
    graph[2][3] = 2.3
    return graph


if __name__ == '__main__':
    g = make_graph()
    v = Viterbi(g)
    v.solve()

# coding=utf-8


class BTreeNode(object):

    def __init__(self, parent=None, yes=None, no=None, label=None):
        self.parent = parent
        self.yes = yes
        self.no = no
        self.label = label


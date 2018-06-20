# coding=utf-8
import os
import sys

join = os.path.join
base = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
base = os.path.dirname(base)
sys.path.append(os.getcwd())
sys.path.append('src/learning_method/tree')

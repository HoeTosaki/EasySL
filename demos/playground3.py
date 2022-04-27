import numpy as np
import torch as tc
import pandas as pd
import networkx as nx
import json
from esl import *
from esl import Utils as eu
import time

def calculate(st,ed):
    c = []
    for idx,ele in enumerate(range(st,ed+1)):
        time.sleep(0.1)
        if idx % 10 == 0:
            print(f'{idx} completed')
        c.append(ele)
    return c

def anal_cal(c):
    print(f'sum of c is {sum(c)}')

if __name__ == '__main__':
    anal_cal(calculate(st=1,ed=100))






import numpy as np
import torch as tc
import pandas as pd
import networkx as nx
import json
from esl import *
from esl import Utils as eu

if __name__ == '__main__':
    a = {'model_name':'Tree-GCN',
         'tree_width':10,'alpha':[0.01,0.02,0.05]}
    b = np.array([[1.2]*10+[0.5]*2]*3)
    c = pd.DataFrame(b,index=['1','2','3'],
                     columns=[f'col{i}' for i in range(12)])
    d = tc.from_numpy(b)

    # save.
    eu.register(a=a, b=b, c=c, d=d)

    # load.
    ddv = eu.ddv()
    a, b, c, d = ddv.a, ddv.b, ddv.c, ddv.d

    # load&print data.
    print(a,'\n',b,'\n',c,'\n',d)




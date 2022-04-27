import json
import h5py
import pickle
import numpy as np
import pandas as pd
import scipy as sc
import dgl
import torch as tc
import networkx as nx

def Xtest_json():
    a1 = {'why':[12,3,4],'why2':12.3}
    a2 = {'why':[12,3,4],'why2':12.3,'why3':{1,2,3}}
    a3 = np.array([1,2,3.5])
    a4 = dgl.DGLGraph()
    a4.add_nodes(5)
    with open('data1.json','w') as f:
        json.dump(a2,f)

    with open('data1.json','r') as f:
        load_dict = json.load(f)
    print(load_dict)

def Xtest_pickle():
    a1 = {'why':[12,3,4],'why2':12.3}
    a2 = {'why':[12,3,4],'why2':12.3,'why3':{1,2,3}}
    a3 = np.array([1,2,3.5])
    a4 = dgl.DGLGraph()
    a4.add_nodes(5)
    a5 = nx.Graph()
    a5.add_edge(1,2)
    a5.add_edge(2,3)
    a = a5
    with open('data2.pkl','wb') as f:
        pickle.dump(a,f)

    with open('data2.pkl','rb') as f:
        load_dict = pickle.load(f)
    print(type(load_dict))
    print(load_dict)

def Xtest_hdf():
    a1 = {'why':[12,3,4],'why2':12.3}
    a2 = {'why':[12,3,4],'why2':12.3,'why3':{1,2,3}}
    a3 = np.array([1,2,3.5])
    a4 = dgl.DGLGraph()
    a4.add_nodes(5)
    a5 = nx.Graph()
    a5.add_edge(1,2)
    a5.add_edge(2,3)
    a = a5



if __name__ == '__main__':
    Xtest_json()
    # Xtest_pickle()

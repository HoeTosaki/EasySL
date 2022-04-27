import networkx as nx

from esl import *
from esl import Utils as eu

import time

def routine1():
    '''
        create a default cluster and directly register some data to ESL.
        Data managed by cluster changed correspondingly along with the running routine after registered.
    '''
    a = [1,2,3,4,5]
    b = {'why':123,'pdj':208}
    print('orginal data: a={},b={}'.format(a,b))

    eu.register(aa=a,bb=b)
    time.sleep(2)
    print('managed data: a={},b={}'.format(eu.inn().aa,eu.inn().bb))

    a[1] = 100
    b['ljk'] = '10086'
    print('modified data: a={},b={}'.format(a,b))
    print('managed data: a={},b={}'.format(eu.inn().aa,eu.inn().bb))

def routine2():
    '''
        read saved data in default data cluster.
    '''
    cls = ESL.from_cluster()
    dv = cls.data_view()
    print('managed data: a={},b={}'.format(dv.inn.aa,dv.inn.bb))


def routine3():
    '''
        separation between different data cluster for the data with the same name.
    '''
    cls1 = ESL.from_cluster(cluster_name='cls1')
    a = [1,2,3,4,5]
    b = [1,2,3]
    cls1.register(var_s=a)
    cls2 = ESL.from_cluster(cluster_name='cls2')
    cls2.register(var_s=b)

    dv1 = cls1.data_view()
    dv2 = cls2.data_view()
    print('cls1: var_s={} | cls2: var_s={}'.format(dv1.inn.var_s,dv2.inn.var_s))

def routine4():
    '''
        different data type.
    :return:
    '''
    a1 = [1,2,3,4,5]
    aa1 = np.array([1,2,3])
    a2 = tc.FloatTensor([1.2,-5.6,7.7,10.1])
    a3 = dgl.DGLGraph()
    a3.add_nodes(4)
    a4 = nx.Graph()
    a4.add_edge(3, 4)
    a4.add_edge(5, 3)
    a4.add_edge(1, 3)
    a5 = pd.DataFrame([[1,2,3],['wy1','why','pdj']],
                      index=['r1','r2'],columns=['c1','c2','c3'])

    cls = ESL.from_cluster(cluster_name='cls_type')
    cls.register(a1=a1,a2=a2,a3=a3,a4=a4,a5=a5)
    cls.register(a1=aa1) # duplicate name of a1 with different type numpy.ndarray

    dv = cls.data_view()
    print(dv.inn.a1)
    print(dv.np.a1)
    print(dv.tc.a2)
    print(dv.dgl.a3)
    print(dv.nx.a4)
    print(dv.pd.a5)

@auto_save
def func_test(in_a,in_b):
    c = [ele +in_a for ele in in_b]
    return [ele*2 for ele in c]

def routine5():
    '''
        define data process function with auto-saving mode, and read input/output data.
    '''
    func_test(in_a=100,in_b=[1,2,3])
    esl_func = ESL.from_func(func=func_test)
    dv = esl_func.data_view()
    print('{}+{}->{}'.format(dv.inn.in_a,dv.inn.in_b,dv.inn.__ret_0__))


@auto_save
# @auto_load
def timable_func(in_a):
    for i in range(in_a):
        print('i:{}'.format(i))
        time.sleep(0.2)
    return 'complete'

def routine6():
    '''
        define data process function with auto-saving mode, and read input/output data.
    '''
    print(timable_func(in_a=10))


if __name__ == '__main__':
    print('hello basic usage.')
    routine1()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # routine2()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # routine3()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # routine4()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # routine5()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # routine6()


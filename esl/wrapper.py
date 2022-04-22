import os
import json
import logging as lg
import numpy as np
import pandas as pd
import json
import torch as tc
import dgl
import networkx as nx

class Wrapper:
    def __init__(self,data,data_name,cluster_pwd):
        self.data = data
        self.data_name = data_name
        self.cluster_pwd=cluster_pwd

    def dump(self):
        raise NotImplementedError

    def save_path(self):
        raise NotImplementedError

    def save_name(self):
        return '{}-{}'.format(self.__class__.type_id(),self.data_name)

    def pwd(self):
        return os.path.join(self.cluster_pwd,self.save_path())

    @classmethod
    def decode_save_name(cls,save_name:str):
        lst = save_name.strip().split('-')
        assert len(lst) == 2
        return lst

    @classmethod
    def load(cls,save_name,cluster_pwd):
        raise NotImplementedError

    @classmethod
    def type_id(cls):
        return 'RAW'

    @classmethod
    def is_support_joint(cls):
        return False

    @classmethod
    def is_support_inline(cls):
        return False


class InnerWrapper(Wrapper):
    def __init__(self,**kwargs):
        super(InnerWrapper, self).__init__(**kwargs)

    def dump(self):
        with open(self.pwd(),'w') as f:
            json.dump(self.data,f)

    def save_path(self):
        return '{}.json'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type,data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = InnerWrapper(data=None,data_name=data_name,cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type,data_name))
            return None
        try:
            with open(wrp.pwd(),'r') as f:
                load_dict = json.load(f)
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        wrp.data = load_dict
        return wrp

    @classmethod
    def type_id(cls):
        return 'inner'



class NumpyWrapper(Wrapper):
    def __init__(self, **kwargs):
        super(NumpyWrapper, self).__init__(**kwargs)

    def dump(self):
        np.save(self.pwd(),self.data)

    def save_path(self):
        return '{}.npy'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type, data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = NumpyWrapper(data=None, data_name=data_name, cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type, data_name))
            return None
        try:
            wrp.data = np.load(wrp.pwd())
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        return wrp

    @classmethod
    def type_id(cls):
        return 'np'


class PandasWrapper(Wrapper):
    def __init__(self, **kwargs):
        super(PandasWrapper, self).__init__(**kwargs)

    def dump(self):
        self.data.to_csv(self.pwd(),index=False)

    def save_path(self):
        return '{}.csv'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type, data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = PandasWrapper(data=None, data_name=data_name, cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type, data_name))
            return None
        try:
            wrp.data = pd.read_csv(wrp.pwd())
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        return wrp

    @classmethod
    def type_id(cls):
        return 'pd'


class DGLWrapper(Wrapper):
    def __init__(self, **kwargs):
        super(DGLWrapper, self).__init__(**kwargs)

    def dump(self):
        dgl.save_graphs(self.pwd(),[self.data])

    def save_path(self):
        return '{}.dgl.graph'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type, data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = DGLWrapper(data=None, data_name=data_name, cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type, data_name))
            return None
        try:
            gs,_ = dgl.load_graphs(wrp.pwd())
            g = gs[0]
            wrp.data = g
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        return wrp

    @classmethod
    def type_id(cls):
        return 'dgl'


class TorchWrapper(Wrapper):
    def __init__(self, **kwargs):
        super(TorchWrapper, self).__init__(**kwargs)

    def dump(self):
        tc.save(self.data,self.pwd())

    def save_path(self):
        return '{}.tensor'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type, data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = TorchWrapper(data=None, data_name=data_name, cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type, data_name))
            return None
        try:
            wrp.data = tc.load(wrp.pwd(),map_location='cpu')
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        return wrp

    @classmethod
    def type_id(cls):
        return 'tc'


class NetworkXWrapper(Wrapper):
    def __init__(self, **kwargs):
        super(NetworkXWrapper, self).__init__(**kwargs)

    def dump(self):
        nx.write_gpickle(self.data, self.pwd())

    def save_path(self):
        return '{}.networkx.graph'.format(self.save_name())

    @classmethod
    def load(cls, save_name, cluster_pwd):
        data_type, data_name = cls.decode_save_name(save_name=save_name)
        assert data_type == cls.type_id()
        wrp = NetworkXWrapper(data=None, data_name=data_name, cluster_pwd=cluster_pwd)
        if not os.path.exists(wrp.pwd()):
            lg.warning('No data found by data type {}, data name {}'.format(data_type, data_name))
            return None
        try:
            wrp.data = nx.read_gpickle(wrp.pwd())
        except IOError:
            lg.warning('data destroyed by data type {}, data name {}'.format(data_type, data_name))
            return None
        return wrp

    @classmethod
    def type_id(cls):
        return 'nx'

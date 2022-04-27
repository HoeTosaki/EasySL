import numpy as np
import pandas as pd
import json
import torch as tc
import dgl
import networkx as nx
import logging as lg
import esl.util as util
import os
import copy
from esl.wrapper import *
from _ctypes import PyObj_FromPtr


class ESL:
    '''
        basic management focused on clustered data.
    '''
    __GLB_ESL_DICT__ = {}
    __ROOT_DIR__ = '.esl_saved'
    __ROOT_DESC__ = 'desc.json'
    def __init__(self,cluster_name='global',cluster_type='GLB',load_data=True,force_delete=False,auto_save=True):
        '''
        :param cluster_name:
        :param cluster_type: 'GLB' for user, other for extended features.
        :param load_data: if data loaded from storage when construct the cluster.
        :param force_delete: force to deleted destroyed data on the disk.
        :param auto_save: auto save data when new data is registered.
        '''
        self.cluster_name = cluster_name
        self.cluster_type = cluster_type
        self.cluster_id = self._name2id()
        self.data_dict = None
        self.data2type = None
        self.init_data_dict()
        self.is_load_data = load_data
        self.force_delete = force_delete
        self.auto_save = auto_save

        if self.cluster_id in self.__class__.__GLB_ESL_DICT__ and self.__class__.__GLB_ESL_DICT__[self.cluster_id] != 'Silence':
            lg.error('detect duplicate cluster instance with the same id {}, plz avoid creating an instance of ESL, '
                     'call ESL.from_cluster instead.'.format(self.cluster_id))
            raise BufferError
        ESL.__GLB_ESL_DICT__[self.cluster_id] = self
        if not os.path.exists(self.pwd()):
            os.mkdir(self.pwd())
        if load_data:
            if not self.load():
                if not self.force_delete:
                    lg.error('ESL {} failed to load some of the data, use option "force_delete=True" to recover.'.format(self.cluster_id))
                    raise IOError
                else:
                    self.dump()
            else:
                self._construct_data2type_dict()

    def _construct_data2type_dict(self):
        if self.data_dict is None:
            return
        self.data2type = {}
        for type in self.data_dict:
            type_data_dict = self.data_dict[type]
            for data_name in type_data_dict:
                if data_name not in self.data2type:
                    self.data2type[data_name] = [type]
                else:
                    self.data2type[data_name].append(type)

    def init_data_dict(self):
        self.data_dict = {}
        self.data_dict['inn'] = {}
        self.data_dict['np'] = {}
        self.data_dict['pd'] = {}
        self.data_dict['dgl'] = {}
        self.data_dict['nx'] = {}
        self.data_dict['tc'] = {}

    @classmethod
    def config_meta_path(cls, save_path=None):
        if save_path is not None:
            cls.__ROOT_DIR__ = os.path.join(save_path,'.esl_saved')
        if not os.path.exists(cls.__ROOT_DIR__):
            os.mkdir(cls.__ROOT_DIR__)
        desc_file = os.path.join(cls.__ROOT_DIR__, cls.__ROOT_DESC__)
        if not os.path.exists(desc_file):
            with open(desc_file,'w') as f:
                json.dump(cls.__GLB_ESL_DICT__,f)

    @classmethod
    def update_meta_info(cls,is_read=True,is_write=True):
        desc_file = os.path.join(cls.__ROOT_DIR__, cls.__ROOT_DESC__)
        if is_read:
            if not os.path.exists(desc_file):
                load_dict = {}
            else:
                try:
                    with open(desc_file, 'r') as f:
                        load_dict = json.load(f)
                except IOError:
                    lg.warning('ESL Storage destroyed or during constructing a new repository.')
                    load_dict = {}
        else:
            load_dict = {}
        if is_write:
            load_dict.update(cls.__GLB_ESL_DICT__)
            cls.__GLB_ESL_DICT__ = copy.deepcopy(load_dict)
            for k in load_dict:
                load_dict[k] = 'Silence'
            with open(desc_file, 'w') as f:
                json.dump(load_dict, f)
        else:
            cls.__GLB_ESL_DICT__ = load_dict

    @classmethod
    def from_cluster(cls,cluster_name='global',cluster_type='GLB',**kwargs):
        cls.config_meta_path()
        intend_id = cls._s_name2id(cluster_name,cluster_type)

        if intend_id in cls.__GLB_ESL_DICT__:
            # pre-search.
            if cls.__GLB_ESL_DICT__[intend_id] != 'Silence':
                return cls.__GLB_ESL_DICT__[intend_id]
            else:
                esl = ESL(cluster_name=cluster_name,cluster_type=cluster_type,**kwargs)
                assert cls.__GLB_ESL_DICT__[intend_id] != 'Silence' # acitivate!
                return esl
        else:
            cls.update_meta_info(is_write=False)
            if intend_id in cls.__GLB_ESL_DICT__:
                assert cls.__GLB_ESL_DICT__[intend_id] == 'Silence'
                esl = ESL(cluster_name=cluster_name,cluster_type=cluster_type,**kwargs)
                assert cls.__GLB_ESL_DICT__[intend_id] != 'Silence'  # acitivate!
                return esl
            else:
                esl = ESL(cluster_name=cluster_name, cluster_type=cluster_type,**kwargs)
                assert intend_id in cls.__GLB_ESL_DICT__ and cls.__GLB_ESL_DICT__[intend_id] != 'Silence'  # acitivate!
                cls.update_meta_info(is_write=True)
                return esl

    @classmethod
    def from_func(cls,func,**kwargs):
        if not hasattr(func,'__name_inner__'):
            lg.warning('from a function that has NOT been decorated by auto_save or auto_sl')
            return None
        return cls.from_cluster(cluster_name=func.__name_inner__,cluster_type='FUNC',**kwargs)

    @classmethod
    def _s_name2id(cls,cluster_name,cluster_type):
        return '{}@{}'.format(cluster_type,cluster_name)

    def pwd(self):
        return os.path.join(self.__class__.__ROOT_DIR__,self.cluster_id)

    def _name2id(self):
        return '{}@{}'.format(self.cluster_type,self.cluster_name)

    def __str__(self):
        return self.cluster_id

    def _check_data_type(self,data):
        if type(data) in [int,float,str,list,dict,set,]:
            return 'inn'
        elif type(data) is np.ndarray:
            return 'np'
        elif type(data) is pd.DataFrame:
            return 'pd'
        elif type(data) is dgl.DGLGraph:
            return 'dgl'
        elif type(data) is nx.Graph:
            return 'nx'
        elif type(data) is tc.Tensor:
            return 'tc'
        else:
            lg.error('ESL encounter unknown data type {}'.format(type(data)))
            raise TypeError

    def register(self, **dict_data):
        '''
            register sub-data in current data cluster.
        '''
        for data_name in dict_data:
            data = dict_data[data_name]
            if type(data) in [int, float, str, list, dict, set,tuple]:
                self.data_dict['inn'][data_name] = (data,False)
            elif type(data) is np.ndarray:
                self.data_dict['np'][data_name] = (data,False)
            elif type(data) is pd.DataFrame:
                self.data_dict['pd'][data_name] = (data,False)
            elif type(data) is dgl.DGLGraph:
                self.data_dict['dgl'][data_name] = (data,False)
            elif type(data) is nx.Graph:
                self.data_dict['nx'][data_name] = (data,False)
            elif type(data) is tc.Tensor:
                self.data_dict['tc'][data_name] = (data,False)
            else:
                lg.error('ESL encountered unknown data type {}, named as {}'.format(type(data),data_name))
                raise TypeError
        if self.auto_save:
            self.dump()
        print(f'{self} all:{self.data_dict}')

    @classmethod
    def data_types(cls):
        return ['inn','np','pd','tc','nx','dgl']

    @classmethod
    def wrap_classes(cls):
        return [InnerWrapper,NumpyWrapper,PandasWrapper,TorchWrapper,NetworkXWrapper,DGLWrapper]

    def dump(self):
        _wrap_classes_ = self.__class__.wrap_classes()
        data_types = self.__class__.data_types()
        # dump cluster info & modified data.
        cluster_dict = {}
        for data_type,_wrap_cls_ in zip(data_types,_wrap_classes_):
            cluster_dict[data_type] = {}
            for data_name in self.data_dict[data_type]:
                data,sgn = self.data_dict[data_type][data_name]
                wrp = _wrap_cls_(data=data, data_name=data_name,cluster_pwd=self.pwd())
                if not sgn:
                    wrp.dump()
                cluster_dict[data_type][wrp.save_name()] = wrp.save_path() # relative path of the specified data.
        with open(os.path.join(self.pwd(),'info.json'), 'w') as f:
            json.dump(cluster_dict, f)

    def load(self):
        if not os.path.exists(os.path.join(self.pwd(),'info.json')):
            return True
        try:
            with open(os.path.join(self.pwd(),'info.json'), 'r') as f:
                load_dict = json.load(f)
        except IOError:
            lg.warning('ESL {} failed to open cluster info'.format(self.cluster_id))
            return False
        assert load_dict is not None
        self.init_data_dict()
        has_error = False
        for data_type,_wrap_class_ in zip(self.__class__.data_types(),self.__class__.wrap_classes()):
            for save_name in load_dict[data_type]:
                wrp = _wrap_class_.load(save_name=save_name,cluster_pwd=self.pwd())
                if wrp is not None:
                    self.data_dict[data_type][wrp.data_name] = (wrp.data,False)
                else:
                    lg.warning('ESL {} detects the destroyed data with save name {}'.format(self.cluster_id,save_name))
                    has_error = True
        return not has_error

    def data_view(self):
        dv = DataView()
        for data_type in self.data_types():
            obj_type = dv.__getattribute__(data_type)
            type_data_dict = self.data_dict[data_type]
            for data_name in type_data_dict:
                setattr(obj_type,data_name,type_data_dict[data_name][0])
        return dv

    def direct_data_view(self):
        if self.data2type is None:
            lg.warning('ESL {} currently NOT support direct data view, plz use data_view() instead.'.format(self.cluster_id))
            return None
        dv = DataView.SubDataView()
        for data_name in self.data2type:
            data_type = self.data2type[data_name]
            if len(data_type) == 1:
                setattr(dv, data_name, self.data_dict[data_type[0]][data_name][0])
            else:
                for i in range(len(data_type)):
                    setattr(dv, '{}_{}'.format(data_name,data_type), self.data_dict[data_type[i]][data_name][0])
        return dv

    def dv(self):
        '''
             alias name for data_view()
        '''
        return self.data_view()

    def ddv(self):
        '''
             alias name for direct_data_view()
        '''
        return self.direct_data_view()

    def inn(self):
        print(f'{self} inn:{self.data_dict}')
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['inn']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

    def np(self):
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['np']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

    def pd(self):
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['pd']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

    def tc(self):
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['tc']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

    def dgl(self):
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['dgl']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

    def nx(self):
        obj = DataView.SubDataView()
        type_data_dict = self.data_dict['nx']
        for data_name in type_data_dict:
            setattr(obj, data_name, type_data_dict[data_name][0])
        return obj

class DataView:
    '''
        Use for return data view of ESL data cluster.
    '''
    def __init__(self):
        self.inn = DataView.SubDataView()
        self.np = DataView.SubDataView()
        self.pd = DataView.SubDataView()
        self.tc = DataView.SubDataView()
        self.dgl = DataView.SubDataView()
        self.nx = DataView.SubDataView()

    class SubDataView:
        pass

'''
Decoration.
'''
def auto_save(func):
    def _decorated_func(**kwargs):
        func_esl = ESL.from_cluster(cluster_name=func.__name__,cluster_type='FUNC',load_data=False,auto_save=False,force_delete=True)
        func_esl.register(**kwargs)
        rets = func(**kwargs)
        if type(rets) is not tuple:
            rets = (rets,)
        ret_dict = {}
        ret_dict['__ret_len__'] = len(rets)
        for idx,ret in enumerate(rets):
            ret_dict['__ret_{}__'.format(idx)] = ret
        func_esl.register(**ret_dict)
        func_esl.dump()
        return tuple(rets) if len(rets) > 1 else rets[0]
    _decorated_func.__name_inner__ = func.__name__
    return _decorated_func

def auto_load(func):
    def _decorated_func(**kwargs):
        func_esl = ESL.from_cluster(cluster_name=func.__name__,cluster_type='FUNC',load_data=True,auto_save=False,force_delete=False)
        dv = func_esl.data_view()
        ret_len = dv.inn.__ret_len__
        ret_list = []
        for idx in range(ret_len):
            cur_id = '__ret_{}__'.format(idx)
            is_found = False
            for data_type in ESL.data_types():
                obj_view =  getattr(dv,data_type)
                if hasattr(obj_view,cur_id):
                    ret_list.append(getattr(obj_view,cur_id))
                    is_found=True
                    break
            if not is_found:
                lg.warning('Not found the insufficient storage of current function.')
                return None
        return tuple(ret_list) if ret_len > 1 else ret_list[0]
    _decorated_func.__name_inner__ = func.__name__
    return _decorated_func

'''
Utils.
'''
class Utils:
    @staticmethod
    def di(var_id):
        return PyObj_FromPtr(var_id)

    @staticmethod
    def register(**dict_data):
        ESL.from_cluster().register(**dict_data)

    @staticmethod
    def data_view():
        return ESL.from_cluster().data_view()

    @staticmethod
    def dv():
        return ESL.from_cluster().dv()

    @staticmethod
    def ddv():
        return ESL.from_cluster().ddv()

    @staticmethod
    def inn():
        return ESL.from_cluster().inn()

    @staticmethod
    def np():
        return ESL.from_cluster().np()

    @staticmethod
    def pd():
        return ESL.from_cluster().pd()

    @staticmethod
    def dgl():
        return ESL.from_cluster().dgl()

    @staticmethod
    def nx():
        return ESL.from_cluster().nx()

if __name__ == '__main__':
    print('hello ESL.')
    esl = ESL()
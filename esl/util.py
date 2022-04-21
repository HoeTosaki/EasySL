from _ctypes import PyObj_FromPtr

def di(var_id):
    return PyObj_FromPtr(var_id)
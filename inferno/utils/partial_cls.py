import functools



def partial_cls(base_cls, name, module, *args, **kwargs):
    return type(name, (base_cls,), {
        '__module__': module,
        '__init__' : functools.partialmethod(base_cls.__init__, *args, **kwargs)
    })

def register_partial_cls(base_cls, name, module, module_dict, *args, **kwargs):
    generatedClass = partial_cls(base_cls=base_cls,name=name, module=module,
        *args, **kwargs)
    module_dict[generatedClass.__name__] = generatedClass
    del generatedClass



if __name__ == "__main__":

    class Conv(object):
        def __init__(self, dim, activation, stride=1):
            pass


    class Conv1D(Conv):
        def __init__(self, activation, stride=1):
            super().__init__(activation=activation,
                dim=1,
                stride=stride)

    class Conv2D(Conv):
        def __init__(self, activation, stride=1):
            super().__init__(activation=activation,
                dim=2,
                stride=stride)

    b = Conv1D(activation='ELU')
    b = Conv2D(activation='ELU')

    for dim in [1,2]:
        register_partial_cls(Conv, 'Conv{}D'.format(dim),__name__, globals(), dim=2)
    
    b = Conv1D(activation='ELU')
    b = Conv2D(activation='ELU')
import functools



def partial_cls(base_cls, name, *args, **kwargs):
    return type(name, (base_cls,), {
        '__init__' : functools.partialmethod(base_cls.__init__, *args, **kwargs)
    })

def register_partial_cls(base_cls, name, module_dict, *args, **kwargs):
    generatedClass = partial_cls(base_cls=base_cls,name=name,
        *args, **kwargs)
    module_dict[generatedClass.__name__] = generatedClass
    del generatedClass
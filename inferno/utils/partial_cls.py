import functools
import sys


def partial_cls(base_cls, name, module, fixed, default):




    def better_partial(f, fixed=None, default=None):
        if fixed is None:
            fixed = {}
        if default is None:
            default = {}
        intersection = fixed.keys() & default.keys()
        if intersection:
            raise TypeError('fixed and default share keys')

        def partial_function(*args, **other_kwargs):
            intersection = fixed.keys() & other_kwargs.keys()
            if len(intersection) == 1:
                raise TypeError("partial function of `{}` got unexpected keyword argument(s) '{}'".format(str(f), str(intersection)))
            combined = {**fixed, **other_kwargs, **default}
            return f(*args, **combined)

        return partial_function



    return type(name, (base_cls,), {
        '__module__': module,
        '__init__' : better_partial(base_cls.__init__, fixed=fixed, default=default),
    })

def register_partial_cls(base_cls, name, module, fixed=None, default=None):
    module_dict = sys.modules[module].__dict__
    generatedClass = partial_cls(base_cls=base_cls,name=name, module=module,
        fixed=fixed, default=default)
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
        register_partial_cls(Conv, 'Conv{}D'.format(dim),__name__, fixed=dict(dim=2))
    
    b = Conv1D(activation='ELU')
    b = Conv2D(activation='ELU')


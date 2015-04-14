import theano.tensor as T
import theano
import numpy as np
import copy

class VariableSet:
    """Holds metadata about variables used in theano functions"""

    def __init__(self, name):
        self.name = name
        self.variables = []
        self.children = []

    def _as_var(self, value, name):
        return theano.shared(value=np.asarray(value, dtype=theano.config.floatX), \
                    name=name, borrow=True)

    def add(self, value, label, low_bound=-np.inf, up_bound=np.inf):
        name = '%s -> %s' % (self.name, label)
        variable = self._as_var(value, name)
        variable.low_bound = low_bound
        variable.up_bound = up_bound
        self.variables.append((variable, value))
        return variable

    def add_child(self, child):
        self.children.append(child)

    def get(self):
        variables = copy.copy(self.variables)
        for c in self.children:
            variables.extend(c.get())
        return variables




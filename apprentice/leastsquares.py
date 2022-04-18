from apprentice.util import Util
from apprentice.function import Function
from apprentice.polyset import PolySet
import numpy as np

class LeastSquares(Function):
    def __init__(self, dim, fnspace, data, s_val, errors, e_val=None, **kwargs):
        super(LeastSquares, self).__init__(dim, fnspace, **kwargs)

        self.data_ = data
        self.err2_ = np.array(errors)**2
        if hasattr(s_val, "vals"):
            self.vals     = s_val.vals
            self.grads    = s_val.grads
            self.hessians = s_val.hessians
        else:
            self.vals     = lambda x: [v.f_x(x)      for v in s_val]
            self.grads    = lambda x: [v.gradient(x) for v in s_val]
            self.hessians = lambda x: [v.hessian(x)  for v in s_val]

        if e_val is not None:
            if hasattr(e_val, "vals"):
                self.evals     = e_val.vals
                self.egrads    = e_val.grads
                self.ehessians = e_val.hessians
            else:
                self.evals     = lambda x: [v.f_x(x)      for v in e_val]
                self.egrads    = lambda x: [v.gradient(x) for v in e_val]
                self.ehessians = lambda x: [v.hessian(x)  for v in e_val]
        else:
            self.evals     = None

    def objective(self, x):

        nom   = ( np.array( self.vals(x) ) - self.data_ )**2
        denom = self.err2_
        if self.evals is not None:
            denom += np.array( self.evals(x) )**2

        return np.sum(nom/denom)

    def gradient(self):
        pass

    def hessian(self):
        pass

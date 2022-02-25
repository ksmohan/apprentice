import numpy as np
from apprentice import space

class Function(object):
    def __init__(self, dim, fnspace=None, **kwargs):
        """
        """
        self.dim_ = dim # raise Exception if no dim given
        if fnspace is None:
            self.fnspace_ = space.Space(dim, [-np.inf for d in range(dim)], [np.inf for d in range(dim)])
        else:
            if isinstance(fnspace, space.Space):
                self.fnspace_ = fnspace
            else:
                try:
                    self.fnspace_ = space.Space(space)
                except Exception as e:
                    print("Unable to interpret space argument:", e)
        self.bounds_ = np.zeros((dim,2))
        for d in range(dim):
            self.bounds[d][0] = self.fnspace_.a_[d]
            self.bounds[d][1] = self.fnspace_.b_[d]

    @classmethod
    def mkEmpty(cls, dim):
        return cls(dim)

    @classmethod
    def fromSpace(cls, spc):
        if isinstance(spc, space.Space):
            return cls(spc.dim, spc)
        else:
            try:
                return cls(len(spc), space.Space.fromList(spc))
            except Exception as e:
                print("Unable to interpret list argument as Space")


    @classmethod
    def fromApproximations(cls, approx):
        """
        approx is a list of approximation objects.
        """
        pass


    @property
    def dim(self):
        return self.dim_

    @property
    def bounds(self):
        return self.bounds_

    @property
    def has_gradient(self):
        """
        Return true if an implementation of gradient is found.
        """
        return hasattr(self, "gradient")

    @property
    def has_hessian(self):
        """
        Return true if an implementation of hessian is found.
        """
        return hasattr(self, "hessian")

    def objective(self,x):
        raise Exception("The function objective must be implemented in the derived class")

    def __call__(self, x):
        return self.objective(x)


class Space(object):
    def __init__(self, dim, a, b, names=None):
        assert(dim == len(a))
        assert(dim == len(b))
        self.dim_ = dim
        self.a_ = a
        self.b_ = b
        self.names_ = names

    @classmethod
    def fromList(cls, lst, names=None):
        return cls(len(lst), [l[0] for l in lst], [l[1] for l in lst], names)

    @property
    def dim(self): return self.dim_

    @property
    def center(self): return [self.a_[d] + 0.5*(self.b_[d]-self.a_[d]) for d in range(self.dim)]

    @property
    def names(self): return self.names_

    def __repr__(self):
        s = "{} dimensional space\n".format(self.dim)
        for d in range(self.dim):
            if self.names is not None:
                s += "{} ".format(self.names[d])

            s += "[{} {}]\n".format(self.a_[d], self.b_[d])

        return s

    def mkSubSpace(self, dims: list[int]):
        """
        Return a Space using only the dimensions specified in dims.
        Useful when fixing parameters.
        """
        newdim = len(dims)
        newnames = [self.names_[d] for d in dims] if not self.names is None else None
        return Space(newdim, [self.a_[d] for d in dims], [self.b_[d] for d in dims], newnames)

    def sample(self, npoints: int, method="uniform", seed=None):
        """
        Sample npoints self.dim_-dimensional pointsrandomly from within this space's bounds.
        Provided methods: uniform,lhs,sobol
        With seed=None, it is guaranteed that successive calls yield different points.
        """

        import numpy as np
        from scipy.stats import qmc
        if method== "uniform":
            if seed is not None:
                np.random.seed(seed)
            points = np.random.uniform(low=self.a_, high=self.b_,size=(npoints, self.dim))
        elif method== "lhs":
            sampler = qmc.LatinHypercube(self.dim, seed=seed)
            sample = sampler.random(n=npoints)
            points = qmc.scale(sample, self.a_, self.b_)
        elif method== "sobol":
            sampler = qmc.Sobol(self.dim, seed=seed)
            sample = sampler.random(n=npoints)
            points = qmc.scale(sample, self.a_, self.b_)
        else:
            raise Exception("Requested sampling method {} not implemented".format(method))

        return sample



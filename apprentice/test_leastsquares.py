import unittest
from apprentice.function import Function
from apprentice.polyset import PolySet
from apprentice.polynomialapproximation import PolynomialApproximation
from apprentice.leastsquares import LeastSquares

from apprentice.util import Util

import numpy as np

def get_rdm_poly(dim, order, n=1):
    import numpy as np
    NC = 2*Util.num_coeffs_poly(dim, order)

    X = np.random.random((NC, dim))

    return [PolynomialApproximation.from_interpolation_points(X, np.random.random(NC), m=np.random.randint(0,order+1)) for _ in range(n)]


class TestFunction(unittest.TestCase):


    def test_fromApproximations(self):
        DIM     =   5
        OMAX    =   5
        NOBJS   = 100
        NPOINTS =  20

        APPR = get_rdm_poly(DIM,OMAX,NOBJS)
        ps = PolySet.from_surrogates(APPR)

        DATA = np.random.random(NOBJS)
        ERRS = np.random.random(NOBJS)

        A = LeastSquares(DIM, None, DATA, APPR, ERRS)
        B = LeastSquares(DIM, None, DATA,   ps, ERRS) ## Should be the faster one


        X = np.random.random((NPOINTS, DIM))
        for i in range(NPOINTS):
            x = X[i]
            VA = A.vals(x)
            VB = B.vals(x)

            for d in range(DIM):
                self.assertAlmostEqual(VA[d], VB[d])


            self.assertAlmostEqual(A(x), B(x))


        # from IPython import embed
        # embed()

if __name__ == "__main__":
    unittest.main()


import unittest
from apprentice.function import Function
from apprentice.polynomialapproximation import PolynomialApproximation

class SimpleFunction(Function):
    def objective(self, x):
        return x**2


class TestFunction(unittest.TestCase):

    def test_empty(self):
        fn = Function.mk_empty(3)
        self.assertEqual(fn.dim,3)


    def test_simple(self):
        sf = SimpleFunction.mk_empty(1)
        self.assertEqual(sf(3), 9)

    def test_fromSpace(self):
        sf = SimpleFunction.from_space( ([0,2],[4,7]))
        self.assertEqual(sf.dim, 2)

    def test_fromApproximations(self):
        APPR = []

        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))
        APPR.append(PolynomialApproximation(2))


        sf = Function.from_surrogates( APPR )
        self.assertEqual(sf.dim, 2)

        with self.assertRaises(Exception) as context:
            sf([0,0])

        self.assertTrue("The function objective must be implemented in the derived class" in str(context.exception))




if __name__ == "__main__":
    unittest.main()

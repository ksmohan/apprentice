import unittest
from function import Function


class SimpleFunction(Function):
    def objective(self, x):
        return x**2


class TestFunction(unittest.TestCase):

    def test_empty(self):
        fn = Function.mkEmpty(3)
        self.assertEqual(fn.dim,3)


    def test_simple(self):
        sf = SimpleFunction.mkEmpty(1)
        self.assertEqual(sf(3), 9)

    def test_fromSpace(self):
        sf = SimpleFunction.fromSpace( ([0,2],[4,7]))
        self.assertEqual(sf.dim, 2)



if __name__ == "__main__":
    unittest.main()

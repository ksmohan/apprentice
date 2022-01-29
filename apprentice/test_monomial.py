import unittest
import numpy as np

from monomial import mono_next_grlex
from monomial import monomialStructure
from monomial import recurrence1D
from monomial import recurrence
from monomial import vandermonde

class TestMonoNextGrlex(unittest.TestCase):
    """
    Test the monomial grlex function
    """
    def test_mono_next_grlex(self):
        self.assertEqual(mono_next_grlex([0,0,0]),[0,0,1])
        self.assertEqual(mono_next_grlex([0,1,0]),[1,0,0])
        self.assertEqual(mono_next_grlex([1,0,0]),[0,0,2])

    def test_monomialStructure(self):
        self.assertEqual(list(monomialStructure(1,2)) , [0,1,2])
        self.assertEqual(list(monomialStructure(2,2).ravel()) , [0,0,0,1,1,0,0,2,1,1,2,0])

        bigstr = monomialStructure(20,4)
        self.assertEqual(bigstr.shape, (10626, 20))

    def test_recurrence1D(self):
        for o in range(4):
            struct = monomialStructure(1,o)
            self.assertEqual(np.sum(recurrence1D(0, struct)), 1)
            self.assertEqual(np.sum(recurrence1D(1, struct)), len(struct))


    def test_recurrence(self):
        for d in range(2,20):
            struct = monomialStructure(d,4)
            self.assertEqual(np.sum(recurrence(np.zeros(d), struct)), 1)
            self.assertEqual(np.sum(recurrence(np.ones(d) , struct)), len(struct))

    def test_vandermonde(self):
        V = vandermonde(np.array([[0,0],[1,1]]), 3)
        struct = monomialStructure(2,3)
        self.assertEqual(list(V[0]), list(recurrence(np.zeros(2), struct)))
        self.assertEqual(list(V[1]), list(recurrence(np.ones(2), struct)))


if __name__ == "__main__":
    unittest.main();

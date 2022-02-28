import unittest
import numpy as np
import pprint

import apprentice
from polynomialapproximation import PolynomialApproximation

class TestPolynomialApproximation(unittest.TestCase):
    @staticmethod
    def get_data(scaled=False):
        X_2D_unscaled = [[5.88130801e-01, 3.71525106e+01],
                         [8.97713728e-01, 3.89955805e+01],
                         [8.91530729e-01, 3.12337372e+01],
                         [8.15837477e-01, 2.35756104e+01],
                         [3.58895856e-02, 3.54050387e+01],
                         [6.91757582e-01, 2.98476208e+01],
                         [3.78680942e-01, 3.26250613e+01],
                         [5.18510945e-01, 3.67899585e+01],
                         [6.57951466e-01, 2.92207879e+01],
                         [1.93850218e-01, 2.99588015e+01],
                         [2.72316402e-01, 3.35882224e+01],
                         [7.18605934e-01, 3.30157183e+01],
                         [7.83003609e-01, 2.53759048e+01],
                         [8.50327640e-01, 2.13464933e+01],
                         [7.75244894e-01, 3.54289028e+01],
                         [3.66643064e-02, 2.96196826e+01],
                         [1.16693735e-01, 2.65841282e+01],
                         [7.51280699e-01, 3.02128211e+01],
                         [2.39218216e-01, 2.52725766e+01],
                         [2.54806014e-01, 2.62102310e+01]]
        X_2D_scaled = [[ -1.,-1.],
                       [ 0.1762616 ,  0.71525106],
                       [ 0.79542746,  0.89955805],
                       [ 0.78306146,  0.12337372],
                       [ 0.63167495, -0.64243896],
                       [-0.92822083,  0.54050387],
                       [ 0.38351516, -0.01523792],
                       [-0.24263812,  0.26250613],
                       [ 0.03702189,  0.67899585],
                       [ 0.31590293, -0.07792121],
                       [-0.61229956, -0.00411985],
                       [-0.4553672 ,  0.35882224],
                       [ 0.43721187,  0.30157183],
                       [ 0.56600722, -0.46240952],
                       [ 0.70065528, -0.86535067],
                       [ 0.55048979,  0.54289028],
                       [-0.92667139, -0.03803174],
                       [-0.76661253, -0.34158718],
                       [ 0.5025614 ,  0.02128211],
                       [-0.52156357, -0.47274234],
                       [-0.49038797, -0.3789769 ],
                       [1.,1.]]

        Y_2D_unscaled = [x[0]**2 + 2*x[0]*x[1] + x[1]**2 for x in X_2D_unscaled]
        Y_2D_scaled = [x[0]**2 + 2*x[0]*x[1] + x[1]**2 for x in X_2D_scaled]
        if scaled:
            return X_2D_scaled,Y_2D_scaled
        else: return X_2D_unscaled,Y_2D_unscaled

    @staticmethod
    def get_exp_pcoeff():
        return [0.,0.,0.,1.,2.,1.]

    @staticmethod
    def get_pa_fit_s1_unscaled():
        (X,Y) = TestPolynomialApproximation.get_data()
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        return P

    def test_from_interpolation_points_s1_unscaled(self):
        P = TestPolynomialApproximation.get_pa_fit_s1_unscaled()
        assert(np.all(np.isclose(P.coeff_numerator, TestPolynomialApproximation.get_exp_pcoeff())))

    def test_from_interpolation_points_s2_unscaled(self):
        (X,Y) = TestPolynomialApproximation.get_data()
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=2)
        assert(np.all(np.isclose(P.coeff_numerator, TestPolynomialApproximation.get_exp_pcoeff())))
    def test_from_interpolation_points_s1_scaled(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        assert(np.all(np.isclose(P.coeff_numerator, TestPolynomialApproximation.get_exp_pcoeff())))
    def test_from_interpolation_points_s2_scaled(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=2)
        assert(np.all(np.isclose(P.coeff_numerator, TestPolynomialApproximation.get_exp_pcoeff())))

    def test_save_and_from_file(self):
        P = TestPolynomialApproximation.get_pa_fit_s1_unscaled()
        tmp_file = "/tmp/pa.json"
        P.save(tmp_file)
        P_from_file = PolynomialApproximation.from_file(tmp_file)
        assert(np.all(np.isclose(P_from_file.coeff_numerator, TestPolynomialApproximation.get_exp_pcoeff())))

    def test_gradient(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        exp_grad = [4.,4.]
        grad = P.gradient([1,1])
        assert(np.all(np.isclose(grad, exp_grad)))

    def test_hessian(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        P = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        exp_hess = [[2.,2.],[2.,2.]]
        hess = P.hessian([1,1])
        assert(np.all(np.isclose(hess, exp_hess)))

    def test_coeff_norm(self):
        P = TestPolynomialApproximation.get_pa_fit_s1_unscaled()
        assert (np.isclose(P.coeff_norm, 4.))


    def test_coeff2_norm(self):
        P = TestPolynomialApproximation.get_pa_fit_s1_unscaled()
        assert (np.isclose(P.coeff2_norm, np.sqrt(6.)))

    def test_f_x(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=False)
        Pusc = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        Psc = PolynomialApproximation.from_interpolation_points(X,Y,
                                                              m=2,
                                                              strategy=1)
        x = [0.5,25]
        expected_value_sc=650.25
        expected_value_usc=0.25902139952
        assert (np.isclose(Pusc(x), expected_value_usc) and np.isclose(Psc(x), expected_value_sc))



    def test_f_X(self):
        (X,Y) = TestPolynomialApproximation.get_data(scaled=True)
        Psc = PolynomialApproximation.from_interpolation_points(X,Y,
                                                                m=2,
                                                                strategy=1)
        X = [[0.5,20],[0.5,25]]
        expected_value_sc=[420.25,650.25]
        assert(np.all(np.isclose(Psc.f_X(X), expected_value_sc)))

if __name__ == "__main__":
    unittest.main()
from apprentice import numba_
from apprentice import *
from apprentice.loader import loadPlugin

from apprentice.objective_base import ObjectiveBase
from apprentice.objectives import *

from apprentice.rationalapproximation   import RationalApproximation
from apprentice.polynomialapproximation import PolynomialApproximation
from apprentice.rationalapproximationSLSQP import RationalApproximationSLSQP
from apprentice.rationalapproximationSIP import RationalApproximationSIP
from apprentice.io import readData, readApprentice, readApprox, readExpData
from apprentice.weights import read_pointmatchers
from apprentice.scaler import Scaler
from apprentice.monomial import monomialStructure
from apprentice.appset import AppSet
from apprentice.functionalapprox import FunctionalApprox
from apprentice.federation import Federation
try:
    from apprentice.GP import GaussianProcess
except ImportError as e:
    print("GPs not available:", e)

"""Numerical checks for the floating-point heat-flow integrand."""

import unittest

from mpmath import mp
from scipy.integrate import quad

from dbn_upper_bound.python.mputility import (
    Ht_complex_integrand as mp_integrand,
)
from dbn_upper_bound.python.utility import Ht_complex_integrand


class TestIntegrand(unittest.TestCase):
    """Compare evaluation with the arbitrary-precision reference."""

    def test_agrees_with_arbitrary_precision(self):
        for u, z, t in [(0, 35+10j, 0.4), (0.3, 35+10j, 0.4),
                        (0.7, 150j, 0.4), (0.2+0.1j, 35+10j, 0.4)]:
            with self.subTest(u=u, z=z, t=t), mp.workdps(64):
                expected = complex(mp_integrand(u, z, t))
                actual = Ht_complex_integrand(u, z, t)
                self.assertAlmostEqual(
                    actual.real, expected.real,
                    delta=max(abs(expected.real)*1e-12, 1e-14))
                self.assertAlmostEqual(
                    actual.imag, expected.imag,
                    delta=max(abs(expected.imag)*1e-12, 1e-14))

    def test_decaying_tail_does_not_overflow(self):
        for z, t in [(0, 32), (150j, 0.4)]:
            with self.subTest(z=z, t=t), mp.workdps(64):
                self.assertEqual(complex(mp_integrand(5, z, t)), 0j)
                self.assertEqual(Ht_complex_integrand(5, z, t), 0j)

    def test_quadrature_with_large_heat_factor(self):
        actual, _ = quad(lambda u: Ht_complex_integrand(u, 0, 32).real,
                         0, 10, epsabs=1e-12)
        with mp.workdps(40):
            expected = float(mp.quad(lambda u: mp_integrand(u, 0, 32).real,
                                    [0, 0.5, 1, 10]))
        self.assertAlmostEqual(actual, expected, delta=1e-12)


if __name__ == '__main__':
    unittest.main()

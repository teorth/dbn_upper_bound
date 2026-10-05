"""Regression checks for removable Riemann-Siegel correction singularities."""
import unittest
from mpmath import mp
from dbn_upper_bound.python.mputility import c0, c1


class TestCorrectionCoefficient(unittest.TestCase):
    """Check correction values and derivatives near both quarter points."""

    def test_removable_singularities(self):
        with mp.workdps(50):
            for point in (mp.mpf('0.25'), mp.mpf('0.75')):
                self.assertEqual(c0(point), mp.mpf('0.5'))
                for delta in (mp.mpf('1e-40'), -mp.mpf('1e-40')):
                    self.assertTrue(mp.almosteq(
                        c0(point + delta), mp.mpf('0.5'),
                        abs_eps=mp.mpf('1e-38')))

    def test_agrees_with_original_away_from_singularities(self):
        with mp.workdps(50):
            for point in ('0', '0.1', '0.5', '0.9', '1'):
                p = mp.mpf(point)
                original = (mp.cos(2 * mp.pi() * (p*p - p - mp.mpf(1)/16))
                            / mp.cos(2 * mp.pi() * p))
                self.assertTrue(mp.almosteq(c0(p), original))

    def test_derivatives_remain_finite_at_quarter_points(self):
        with mp.workdps(50):
            for point in (mp.mpf('0.25'), mp.mpf('0.75')):
                expected = (-mp.diff(c0, point + mp.mpf('1e-20'), 3)
                            / (96 * mp.pi()**2))
                self.assertTrue(mp.almosteq(
                    c1(point), expected, abs_eps=mp.mpf('1e-18')))

import unittest
from mpmath import mp

from dbn_upper_bound.python.mputility import composite_trapezoid, vwf_err, vwf_err_old


def drifted_trap(lower, upper, step):
    """Previous loop: half-weight only when a node compares equal to an endpoint."""
    total = mp.mpf(0)
    sigma = lower
    while sigma <= upper:
        weight = mp.mpf('0.5') if sigma == lower or sigma == upper else mp.mpf(1)
        total += weight
        sigma += step
    return total * step


class TestTrapezoid(unittest.TestCase):
    def test_binary_step_still_integrates_a_constant(self):
        # From -10 to 10, adding binary 0.01 includes a last node that is not
        # equal to 10. Equality half-weighting therefore misses it.
        width = mp.mpf(20)
        got = composite_trapezoid(-10, 10, 0.01, lambda sigma: mp.mpf(1))
        self.assertTrue(mp.almosteq(got, width, abs_eps=mp.mpf('1e-12')))
        self.assertFalse(mp.almosteq(drifted_trap(-10, 10, 0.01), width, abs_eps=mp.mpf('1e-6')))

    def test_optimized_error_integral_matches_the_reference(self):
        s = mp.mpc('0.5', '100')
        t = mp.mpf('0.4')
        with mp.workdps(25):
            self.assertTrue(mp.almosteq(
                vwf_err(s, t, lim=1, h=mp.mpf('0.5')),
                vwf_err_old(s, t, lim=1, h=mp.mpf('0.5'))))


if __name__ == '__main__':
    unittest.main()

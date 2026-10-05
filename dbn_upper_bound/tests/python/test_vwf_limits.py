import unittest
from mpmath import mp
from dbn_upper_bound.python.mputility import vwf_err, vwf_err_old


class TestErrorIntegrationLimits(unittest.TestCase):
    def test_fractional_integration_limit_matches_reference(self):
        s = mp.mpc('0.5', '100')
        t = mp.mpf('0.4')
        for limit in (2.5, mp.mpf('2.5'), mp.mpf('2')):
            with self.subTest(limit=limit):
                expected = vwf_err_old(s, t, lim=limit, h=mp.mpf('0.25'))
                actual = vwf_err(s, t, lim=limit, h=mp.mpf('0.25'))
                self.assertTrue(mp.almosteq(actual, expected))

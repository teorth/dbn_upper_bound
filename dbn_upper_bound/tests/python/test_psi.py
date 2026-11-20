import unittest
from mpmath import mp

from dbn_upper_bound.python.mputility import psi


def direct_psi(p):
    """The original quotient, which is wrong exactly at half-integers."""
    p = mp.mpc(p)
    trig = (mp.cos(mp.pi() * (p * p / 2 - p - mp.mpf('1') / 8))
            / mp.cos(mp.pi() * p))
    phase = mp.exp(0.5j * mp.pi() * p * p - 5j * mp.pi() / 8)
    return 2 * mp.pi() * trig * phase


class TestPsi(unittest.TestCase):
    def test_half_integers_match_the_limit(self):
        with mp.workdps(40):
            expected = {
                '0.5': -mp.pi() * 1j,
                '-0.5': -3 * mp.pi() * 1j,
                '1.5': mp.pi() * 1j,
                '2.5': 3 * mp.pi() * 1j,
            }
            for point, limit in expected.items():
                with self.subTest(point=point):
                    self.assertTrue(mp.almosteq(psi(mp.mpf(point)), limit))
                    neighbor = psi(mp.mpf(point) + mp.mpf('1e-10'))
                    self.assertTrue(mp.almosteq(neighbor, limit, abs_eps=mp.mpf('1e-8')))
                    # the unsimplified quotient is not that limit
                    self.assertFalse(mp.almosteq(direct_psi(mp.mpf(point)), limit, abs_eps=mp.mpf('1e-6')))

    def test_agrees_with_original_away_from_half_integers(self):
        with mp.workdps(30):
            for point in ('0.25', '0.1', '1.2', '-0.2'):
                with self.subTest(point=point):
                    self.assertTrue(mp.almosteq(psi(mp.mpf(point)), direct_psi(point)))
            z = mp.mpc('0.3', '0.2')
            self.assertTrue(mp.almosteq(psi(z), direct_psi(z)))


if __name__ == '__main__':
    unittest.main()

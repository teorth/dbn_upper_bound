# Contour plots for H/B0 and (H'/H). Helpers match
# Ht_small_x_fixed_mesh_verification.py so this file runs standalone.

from mpmath import mp

mp.dps = 30
mp.pretty = True

def mpf(x):
    return mp.mpf(x)

def log(n):
    return mp.log(n)

def exp(n):
    return mp.exp(n)

def sqrt(x):
    return mp.sqrt(x)

def gamma(z):
    return mp.gamma(z)

def cos(z):
    return mp.cos(z)

def conj(a):
    return a.conjugate()

def sum(n, N, summand):
    return mp.nsum(summand, [n, N])

Pi = mp.pi()
I = 1j

def alpha1(s):
    return 1 / (2 * s) + 1 / (s - 1) + (1 / 2) * log(s / (2 * Pi))

def H01(s):
    return (1 / 2) * s * (s - 1) * Pi ** (-s / 2) * sqrt(2 * Pi) * exp((s / 2 - 1 / 2) * log(s / 2) - s / 2)

def C0(p):
    return (exp(Pi * I * (p ** 2 / 2 + 3 / 8)) - I * sqrt(2) * cos(Pi * p / 2)) / (2 * cos(Pi * p))

def B0_eff(x, y=0.4, t=0.4):
    return (1 / 8) * exp((t / 4) * alpha1((1 + y - I * x) / 2) ** 2) * H01((1 + y - I * x) / 2)

def abceff_x(x, y=0.4, t=0.4):
    T = x / 2
    Tdash = T + Pi * t / 8
    a = sqrt(Tdash / (2 * Pi))
    N = mp.floor(a)
    p = 1 - 2 * (a - N)
    U = exp(-I * ((Tdash / 2) * log(Tdash / (2 * Pi)) - Tdash / 2 - Pi / 8))
    sig = (1 - y) / 2
    s = sig + I * T
    sdash = sig + I * Tdash
    alph1 = alpha1(s)
    alph2 = alpha1(1 - s)
    A0 = exp((t / 4) * alph1 ** 2) * H01(s)
    B0 = exp((t / 4) * alph2 ** 2) * H01(1 - s)
    A_sum = sum(1, N, lambda n: n ** ((t / 4.0) * log(n) - (t / 2.0) * alph1 - s))
    B_sum = sum(1, N, lambda n: n ** ((t / 4.0) * log(n) - (t / 2.0) * alph2 - (1 - s)))
    A = A0 * A_sum
    B = B0 * B_sum
    termC1 = Pi ** (-sdash / 2) * gamma(sdash / 2) * (a ** (-sig)) * C0(p) * U
    termC2 = Pi ** (-(1 - sdash) / 2) * gamma((1 - sdash) / 2) * (a ** (-(1 - sig))) * conj(C0(p)) * conj(U)
    C = exp(t * Pi ** 2 / 64) * (sdash * (sdash - 1) / 2) * ((-1) ** N) * (termC1 + termC2)
    return (A + B - C) / 8

def newton_quot_abc(x, y=0.4, t=0.4, h=0.000001):
    return (abceff_x(x + h, y, t) - abceff_x(x, y, t)) / h

import matplotlib.pyplot as plt
import numpy as np

#H/B0 plot

C1 = [abceff_x(x,y=0.45)/B0_eff(x,y=0.45) for x in [a/100.0 for a in range(1,30000)]]
C2 = [abceff_x(300,y)/B0_eff(300,y) for y in [a/1000.0 for a in reversed(range(400,450))]]
C3 = [abceff_x(x,y=0.4)/B0_eff(x,y=0.4) for x in [a/100.0 for a in reversed(range(1,30000))]]
C4 = [abceff_x(0,y)/B0_eff(0,y) for y in [a/1000.0 for a in range(400,450)]]
C=C1+C2+C3+C4
X = [x.real for x in C]
Y = [x.imag for x in C]
maxint_XY = int(mp.ceil(max(max(X),max(Y))))
range_axis = [a/100.0 for a in range(-100*maxint_XY,100*maxint_XY)]
sz=0.01
plt.scatter(X,Y, color='red', s=sz)
plt.scatter([0 for i in range_axis],[i for i in range_axis], color='blue', s=sz)
plt.scatter([i for i in range_axis],[0 for i in range_axis], color='blue', s=sz)

plt.show()


#H'/H plot

LC1 = [newton_quot_abc(x,0.45)/abceff_x(x,0.45) for x in [a/100.0 for a in range(1,30000)]]
LC2 = [newton_quot_abc(300,y)/abceff_x(300,y) for y in [a/1000.0 for a in reversed(range(400,450))]]
LC3 = [newton_quot_abc(x,0.4)/abceff_x(x,0.4) for x in [a/100.0 for a in reversed(range(1,30000))]]
LC4 = [newton_quot_abc(0,y)/abceff_x(0,y) for y in [a/1000.0 for a in range(400,450)]]
LC=LC1+LC2+LC3+LC4
X = [x.real for x in LC]
Y = [x.imag for x in LC]
maxint_XY1 = int(mp.ceil(max(max(X),max(Y))))
maxint_XY2 = int(mp.ceil(max(max(-Xi for Xi in X),max(-Yi for Yi in Y))))
range_axis = [a/100.0 for a in range(-100*maxint_XY2,25*maxint_XY)]
sz=0.01
plt.scatter(X,Y, color='red', s=sz)
plt.scatter([0 for i in range_axis],[i for i in range_axis], color='blue', s=sz)
plt.scatter([i for i in range_axis],[0 for i in range_axis], color='blue', s=sz)

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:58:49 2026

@author: chad-briddon (assisted by Claude - Opus 5.5)

Library of Jacobi elliptic functions.

Conventions
-----------
All functions take the *parameter* m (= k^2, the SciPy convention), not the
modulus k used in Appendix A of the paper and in DLMF. The amplitude is
phi = am(u, m).

Definitions are sourced from "Handbook of mathematical functions with formulas,
graphs, and mathematical tables" (A&S) and from the NIST Digital Library of
Mathematical Functions (DLMF) sections 19.2, 19.16, and 19.25
('https://dlmf.nist.gov'). Equations from 'Exact analytic solutions for the
n=1 chameleon field in planar systems with an arbitrary number of uniform
layers' are labelled (B&P).

Numerical notes
---------------
* Sn, Ns and Sc are evaluated without the subtractive cancellations of the
  textbook forms:
      Sn(u, m) = (u - E(phi, m))/m        loses ~ -log10(m) digits as m -> 0,
      Sc(u, m) = (dn*sc - E(phi, m))/(1-m) is 0/0 as m -> 1.
  Instead they are written in terms of Carlson's integral R_D:
      Sn(u, m) = (sn^3/3) R_D(cn^2, dn^2, 1)     (used for m < 1/2),
      Sc(u, m) = (sn^3/3) R_D(dn^2, 1, cn^2)     (used for all m).
  To derive the new expression for Sn we first note that from its definition in
  DLMF Eq. (19.16.5), we have R_D(x/s^2, y/s^2, z/s^2) = s^3 R_D(x, y, z).
  Next we combine this identity with DLMF Eqs. (19.2.6) & (19.25.13), and
  A&S Eq. (16.26.1). For the derivation of Sc we use the above identity with
  DLMF Eq. (19.25.11) and A&S Eq. (16.26.9). Note that for both derivations
  DLMF writes Eqs. (19.25.11) & (19.25.13) with c = csc^2(phi) = 1/sn^2
  as the variable.
* sn, cn, dn are taken directly from scipy.special.ellipj (not via
  cos(am u), which loses digits as am u -> pi/2).
* dc - 1 and nc - 1 are provided in cancellation-free form, for use in
  differences such as dc(u_R) - dc(u_L) when m is close to 1.
* Sn_Inverse and Ns_Inverse use a bracketed Newton iteration (dSn/du = sn^2,
  dNs/du = ns^2), falling back to bisection whenever a Newton step leaves the
  bracket.

AI declaration (model used - Claude [Opus 5.5]):
    The expressions defined using R_D were recommended by Claude.
    Some of the functions below were also written by Claude.

"""

import numpy as np

from scipy.special import ellipj
from scipy.special import ellipk as K
from scipy.special import ellipkinc as F
from scipy.special import ellipeinc as E
from scipy.special import elliprd as R_D  # Requires SciPy >= 1.8


' --------------------------------------------------------------------------- '
' ------------------------ Jacobi elliptic functions ------------------------ '
' --------------------------------------------------------------------------- '


def sn(u, m):
    "Source : A&S Eq. (16.1.5)."
    return ellipj(u, m)[0]


def ns(u, m):
    "Source : A&S Eq. (16.3.1)."
    return 1/ellipj(u, m)[0]


def nc(u, m):
    "Source : A&S Eq. (16.3.2)."
    return 1/ellipj(u, m)[1]


def sd(u, m):
    "Source : A&S Eq. (16.3.2)."
    sn_, dn_ = ellipj(u, m)[::2]
    return sn_/dn_


def dc(u, m):
    "Source : A&S Eq. (16.3.1)."
    cn_, dn_ = ellipj(u, m)[1:3]
    return dn_/cn_


def dc_minus_1(u, m, mc=None):
    '''
    dc(u, m) - 1 without cancellation, using dn^2 - cn^2 = (1 - m) sn^2:
        dc - 1 = (1 - m) sn^2 / (cn (cn + dn)).
    Use for differences dc(u_R) - dc(u_L) when m -> 1 (|C| >> 1).
    If the complementary parameter mc = 1 - m is known exactly (e.g.
    mc = 1/C^2 in the |C| >= 1 branch), pass it: computing 1 - m from a
    rounded m loses ~ -log10(1 - m) digits.
    '''
    if mc is None:
        mc = 1 - m
    sn_, cn_, dn_ = ellipj(u, m)[:3]
    return mc*sn_**2/(cn_*(cn_ + dn_))


def nc_minus_1(u, m):
    '''
    nc(u, m) - 1 without cancellation, using 1 - cn^2 = sn^2:
        nc - 1 = sn^2 / (cn (1 + cn)).
    '''
    sn_, cn_ = ellipj(u, m)[:2]
    return sn_**2/(cn_*(1 + cn_))


def sn_Inverse(u, m):
    "Source : A&S Eq. (16.1.5)."
    return F(np.arcsin(u), m)


def ns_Inverse(u, m):
    "Source : Using both A&S Eqs. (16.3.1) & (16.1.5)."
    return F(np.arcsin(1/u), m)


' -------------------------------------------------------------------------- '
' -------------------------- Integrals of squares -------------------------- '
' -------------------------------------------------------------------------- '


def Sn(u, m):
    '''
    Sn(u, m) = int_0^u sn^2(t, m) dt.
    For m < 1/2 the textbook form (u - E(am u, m))/m [A&S Eq. (16.26.1)],
    which loses ~ -log10(m) digits as m -> 0, is replaced by
    (sn^3/3) R_D(cn^2, dn^2, 1). For m >= 1/2 the textbook form has no
    cancellation and is the more accurate of the two near u = K.
    '''
    if m >= 0.5:
        ph = ellipj(u, m)[3]
        return (u - E(ph, m))/m
    sn_, cn_, dn_ = ellipj(u, m)[:3]
    return (sn_**3/3)*R_D(cn_*cn_, dn_*dn_, 1.0)


def Ns(u, m):
    '''
    Ns(u, m) = u - E(am u, m) - cn dn/sn [A&S Eq. (16.26.10)], using the
    identity u - E(am u, m) = m Sn(u, m) [A&S Eq. (16.26.1)] so the first two
    terms do not cancel.
    '''
    sn_, cn_, dn_ = ellipj(u, m)[:3]
    return m*Sn(u, m) - cn_*dn_/sn_


def Sc(u, m, mc=None):
    '''
    Sc(u, m) = int_0^u sc^2(t, m) dt.
    Textbook form (dn sc - E(am u, m))/(1 - m) [A&S Eq. (16.26.9)] rewritten
    as (sn^3/3) R_D(dn^2, 1, cn^2) as derived above. This expression is finite
    and accurate as m -> 1 (no 0/0).

    Optionally pass mc = 1 - m exactly, then dn^2 is evaluated as
    cn^2 + mc sn^2.
    '''
    sn_, cn_, dn_ = ellipj(u, m)[:3]
    dn2 = dn_*dn_ if mc is None else cn_*cn_ + mc*sn_*sn_
    return (sn_**3/3)*R_D(dn2, 1.0, cn_*cn_)


' --------------------------------------------------------------------------- '
' --------------------------------- Inverses -------------------------------- '
' --------------------------------------------------------------------------- '


def _increasing_inverse(f, df, y, a, b, xtol=1e-15, maxiter=200):
    '''
    Solve f(x) = y for x in (a, b), where f is strictly increasing.
    Bracketed Newton iteration: each Newton step that would leave the current
    bracket (or is not finite) is replaced by a bisection step. Works
    element-wise on arrays.
    '''
    y = np.asarray(y, dtype=float)
    lo = np.full(y.shape, float(a))
    hi = np.full(y.shape, float(b))
    x = 0.5*(lo + hi)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for _ in range(maxiter):
            g = f(x) - y
            lo = np.where(g < 0, x, lo)
            hi = np.where(g >= 0, x, hi)

            x_new = x - g/df(x)
            bad = ~np.isfinite(x_new) | (x_new <= lo) | (x_new >= hi)
            x_new = np.where(bad, 0.5*(lo + hi), x_new)

            converged = (np.abs(x_new - x) <= xtol*np.maximum(1.0, np.abs(x))
                         ) | (hi - lo <= xtol*np.maximum(1.0, np.abs(x)))
            x = x_new
            if np.all(converged):
                break

    return x


def Sn_Inverse(y, m):
    '''
    Inverse of Sn(u, m) on 0 <= u <= K(m).
    Values outside the range are clipped so y <= 0 returns 0 and
    y >= Sn(K(m), m) returns K(m). This gives x_c = 0 in B&P Eq. (17) when the
    maximum cannot lie inside the region.
    '''
    scalar = np.ndim(y) == 0
    y = np.atleast_1d(np.asarray(y, dtype=float))
    K_m = K(m)
    Sn_K = Sn(K_m, m)

    out = np.empty_like(y)
    low, high = y <= 0, y >= Sn_K
    out[low], out[high] = 0.0, K_m

    mid = ~(low | high)
    if np.any(mid):
        out[mid] = _increasing_inverse(lambda u: Sn(u, m),
                                       lambda u: ellipj(u, m)[0]**2,
                                       y[mid], 0.0, K_m)

    return out[0] if scalar else out


def Ns_Inverse(y, m):
    '''
    Inverse of Ns(u, m) on 0 < u <= K(m) (Ns -> -infinity as u -> 0).
    Values outside the range are clipped so y >= Ns(K(m), m) return K(m).
    '''
    scalar = np.ndim(y) == 0
    y = np.atleast_1d(np.asarray(y, dtype=float))
    K_m = K(m)
    Ns_K = Ns(K_m, m)

    out = np.empty_like(y)
    high = y >= Ns_K
    out[high] = K_m

    rest = ~high
    if np.any(rest):
        out[rest] = _increasing_inverse(lambda u: Ns(u, m),
                                        lambda u: 1/ellipj(u, m)[0]**2,
                                        y[rest], 0.0, K_m)

    return out[0] if scalar else out

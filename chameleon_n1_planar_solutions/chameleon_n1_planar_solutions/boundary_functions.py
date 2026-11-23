#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:58:49 2026

@author: chad-briddon

Library of functions used to calculate the boundary values of an N-layer
planar system.

Equation referenced are from 'Exact analytic solutions for the n=1 chameleon
field in planar systems with an arbitrary number of uniform layers'.

"""

import numpy as np

from scipy.special import ellipj, ellipkinc as F
from chameleon_n1_planar_solutions.elliptic_functions import (
    K, sn, ns, sn_Inverse, ns_Inverse, Sn, Ns, Sc, Sn_Inverse, Ns_Inverse,
    dc_minus_1, nc_minus_1)


' -------------------------------------------------------------------------- '
' ------------ Functions for transformsation used in interior() ------------ '
' -------------------------------------------------------------------------- '


def w_of_y(y, C_abs, nu2):
    "C^2 < 1 branch, replaces sn_Inverse((y-1)/sqrt((y-1)**2 + 4*y*C2), nu2)."
    return F(np.arctan2(y - 1, 2*C_abs*np.sqrt(y)), nu2)


def y_of_w(w, C_abs, nu2):
    "C^2 < 1 branch, replaces 2/(1 - C_abs*sd(w, nu2)) - 1."
    sn_, cn_, dn_ = ellipj(w, nu2)[:3]
    r = (dn_ + C_abs*np.abs(sn_))/cn_
    return np.where(sn_ >= 0, r**2, r**-2)


def u_of_y(y, mu2):
    "C^2 >= 1 branch, replaces sn_Inverse((y-1)/(y+1), mu2)."
    return F(np.arctan2(y - 1, 2*np.sqrt(y)), mu2)


def y_of_u(u, mu2):
    "C^2 >= 1 branch, replaces 2/(1 - sn(u, mu2)) - 1."
    sn_, cn_ = ellipj(u, mu2)[:2]
    r = (1 + np.abs(sn_))/cn_
    return np.where(sn_ >= 0, r**2, r**-2)


def G(x):
    return x - np.arctanh(np.minimum(x, 1/x))


def extremum_test(field_boundary_L, field_boundary_R, field_VEV, L_RS):

    U_L = np.sqrt(np.asarray(field_boundary_L, dtype=float)/field_VEV)
    U_R = np.sqrt(np.asarray(field_boundary_R, dtype=float)/field_VEV)

    with np.errstate(divide='ignore', invalid='ignore'):
        same_side = (U_L - 1)*(U_R - 1) > 0
        L_sep_rs = np.where(same_side, np.abs(G(U_L) - G(U_R)), np.inf)
    return L_sep_rs < L_RS


' --------------------------------------------------------------------------- '
' ------ Functions used to calculate k and C, and find boundary values ------ '
' --------------------------------------------------------------------------- '


def parameter_residual(gamma, data):
    '''
    Given guess for curve parameter gamma (placeholder for k and |C|) compute
    the residual of the relevent constraint equation.
    See Eqs. (19), (29), (39), and (42).

    Parameters
    ----------
    gamma : float
        Placeholder for curve parameters k and |C|. In extrema case 'k = gamma'
        and in the VEV case '|C| = (1 - gamma)/gamma'. In either case the range
        of gamma is (0, 1).
    data : tuple
        Constant parameters needed for solver. Entries are:
            data[0] - L_rs = 0.5*'region length'/'compton wavelength in region'
            data[1] - field VEV in region
            data[2] - field value at left boundary
            data[3] - field value at right boundary
            data[4] - bool value for "is region in extremum scenario".

    Returns
    -------
    residual : float
        Residual of corresponding constarin function (want to be zero).
    is_monotonic : bool
        Statement of whether the curve is monotonic in this region.

    '''

    L_rs, field_VEV, field_boundary_L, field_boundary_R, is_extremum = data

    if is_extremum:

        if max(field_boundary_L, field_boundary_R) < field_VEV:
            ' ---- Scenario : extrema (maximum) ---- '

            k = gamma

            k2 = k**2
            L_mod = L_rs/np.sqrt(k**3)

            sqrt_x_L = np.sqrt(field_boundary_L/(field_VEV*k))
            sqrt_x_R = np.sqrt(field_boundary_R/(field_VEV*k))

            # Calculate x_c.
            Sn_K = Sn(K(k2), k2)
            sqrt_x_c = sn(Sn_Inverse(Sn_K - L_mod, k2), k2)

            # Calculate residual [Eq. (19)].
            if min(sqrt_x_L, sqrt_x_R) > sqrt_x_c:
                # Maximum occurs inside region.
                is_monotonic = False
                residual = abs(2*Sn_K -
                               Sn(sn_Inverse(sqrt_x_L, k2), k2) -
                               Sn(sn_Inverse(sqrt_x_R, k2), k2)) - L_mod
            else:
                # Maximum occurs outside region (is fictitious).
                is_monotonic = True
                residual = L_mod - abs(Sn(sn_Inverse(sqrt_x_R, k2), k2) -
                                       Sn(sn_Inverse(sqrt_x_L, k2), k2))

        elif min(field_boundary_L, field_boundary_R) > field_VEV:
            ' ---- Scenario : extrema (minimum) ---- '

            k = gamma

            g2 = 1/k**2  # use g := 1/k.
            L_mod = L_rs/np.sqrt(k)

            x_L = field_boundary_L/(field_VEV*k)
            x_R = field_boundary_R/(field_VEV*k)

            sqrt_x_L = np.sqrt(x_L)
            sqrt_x_R = np.sqrt(x_R)

            # Calculate x_c.
            Ns_K = Ns(K(g2), g2)
            sqrt_x_c = ns(Ns_Inverse(Ns_K - L_mod, g2), g2)

            # Calculate residual [Eq. (29)].
            if max(sqrt_x_L, sqrt_x_R) < sqrt_x_c:
                # Minimum occurs inside region.
                is_monotonic = False
                residual = L_mod - abs(2*Ns_K -
                                       Ns(ns_Inverse(sqrt_x_L, g2), g2) -
                                       Ns(ns_Inverse(sqrt_x_R, g2), g2))
            else:
                # Minimum occurs outside region (is fictitious).
                is_monotonic = True
                residual = abs(Ns(ns_Inverse(sqrt_x_R, g2), g2) -
                               Ns(ns_Inverse(sqrt_x_L, g2), g2)) - L_mod

    else:
        ' ---- Scenario : passes VEV ---- '

        is_monotonic = True

        C_abs = (1/gamma) - 1
        C2 = C_abs**2

        y_L = field_boundary_L/field_VEV
        y_R = field_boundary_R/field_VEV

        if C_abs >= 1:
            ' ---- Scenario : passes VEV (C^2 >= 1) ---- '

            mu2 = 1 - 1/C2

            u_L = u_of_y(y_L, mu2)
            u_R = u_of_y(y_R, mu2)

            # Calculate residual [Eq. (39)].
            A = (u_R - u_L) + 2*(Sc(u_R, mu2, mc=1/C2) - Sc(u_L, mu2, mc=1/C2))
            B = dc_minus_1(u_R, mu2, mc=1/C2) - dc_minus_1(u_L, mu2, mc=1/C2)
            residual = abs(A/(2*C_abs) + C_abs*B) - L_rs

        else:
            ' ---- Scenario : passes VEV (C^2 < 1) ---- '

            nu2 = 1 - C2

            w_L = w_of_y(y_L, C_abs, nu2)
            w_R = w_of_y(y_R, C_abs, nu2)

            # Calculate residual [Eq. (42)].
            A = (w_R - w_L)/2 + C2*(Sc(w_R, nu2, mc=C2) - Sc(w_L, nu2, mc=C2))
            B = nc_minus_1(w_R, nu2) - nc_minus_1(w_L, nu2)
            residual = abs(A + C_abs*B) - L_rs

    return residual, is_monotonic


def solve_for_parameter(data):
    '''
    Use bisection method to find curve parameter that minimises the size of
    the residual given by 'parameter_residual()'.

    Parameters
    ----------
    data : tuple
        Constant parameters needed for solver. Entries are:
            data[0] - L_rs = 0.5*'region length'/'compton wavelength in region'
            data[1] - field VEV in region
            data[2] - field value at left boundary
            data[3] - field value at right boundary
            data[4] - bool value for "is region in extremum scenario".

    Returns
    -------
    solution_type : list of string
        First entry gives type of solution returned ('extrema' or 'VEV').
        Second enrty gives sub-type if applicable. Third entry is bool for
        statement "is curve monotonic in this region" (True if any extremum is
        fictitious, or for the VEV case). Example ['extrema', 'maximum', False]
        means an extremum scenario with a real maximum inside the region.
    curve_parameter : float
        Placeholder for curve parameters k and |C|. In extrema case is 'k' and
        in the VEV case is '1/(1 + |C|)'. In either case the range of
        'curve_parameter' is (0, 1).

    '''

    solution_type = [None, None, None]
    L_rs, field_VEV, field_boundary_L, field_boundary_R, is_extremum = data

    # Make initial guess based on scenario.
    if is_extremum:

        if max(field_boundary_L, field_boundary_R) < field_VEV:
            ' ---- Scenario : extrema (maximum) ---- '
            a, b = max(field_boundary_L, field_boundary_R)/field_VEV, 1
            solution_type[0] = "extrema"
            solution_type[1] = "maximum"

        elif min(field_boundary_L, field_boundary_R) > field_VEV:
            ' ---- Scenario : extrema (minimum) ---- '
            a, b = 1, min(field_boundary_L, field_boundary_R)/field_VEV
            solution_type[0] = "extrema"
            solution_type[1] = "minimum"

    else:
        ' ---- Scenario : passes VEV ---- '
        a, b = 0, 1
        solution_type[0] = "VEV"

    # Numerically solve for the curve parameter using bisection method.
    # Automaticlly terminates after 2000 iterations.
    for _ in range(2000):

        c = (a + b)/2

        if parameter_residual(c, data)[0] > 0:
            b = c
        else:
            a = c

        if abs(b - a) < (1e-14)*b:
            break

    curve_parameter = (a + b)/2

    _, is_monotonic = parameter_residual(curve_parameter, data)
    solution_type[2] = is_monotonic

    return solution_type, curve_parameter


def residual_boundary_constraint(boundary_values, *setup):
    '''
    For given set of boundary values, calculates the residual of each boundary.
    The residual is calculates using Eq. (50) and the curve parameters obtained
    from 'solve_for_parameter()' given the initial boundary values. If the
    boundary values are correct then the difference is (idealistically) zero.

    To be used with 'scipy.optimize.least_squares()' to find correct boundary
    values.

    Parameters
    ----------
    boundary_values : numpy.array
        Values of the field at the boundaries of neighbouring regions.
    *setup : tuple
        Tuple containing two numpy.arrays, with each entry in the arrays
        corresponding to a different region (going from left to right).
        First array contains VEV values while second the array contains the
        values of 'L_rs = 0.5*length/compton_wavelength'.

    Returns
    -------
    numpy.array
        Array of calculated residual values for each region.

    '''

    field_VEVs, L_RSs = setup

    field_boundary_L = boundary_values[:-1]
    field_boundary_R = boundary_values[1:]

    is_extremum = extremum_test(field_boundary_L, field_boundary_R,
                                field_VEVs[1:-1], L_RSs)

    # Compute delta coefficents for each region.
    deltas = [0]  # First entry corresponds to exterior region.
    solution_types = [["exterior"]]

    for phi_L, phi_R, field_VEV, L_rs, ie in zip(field_boundary_L,
                                                 field_boundary_R,
                                                 field_VEVs[1:], L_RSs,
                                                 is_extremum):

        # Calculate correct curve parameter for given boundary values.
        solution_type, para = solve_for_parameter(data=(L_rs, field_VEV,
                                                        phi_L, phi_R, ie))

        solution_types.append(solution_type)

        if solution_type[0] == "extrema":
            deltas.append(((1 - para)**2)/para)

        else:
            deltas.append(-4*((1 - para)/para)**2)

    deltas.append(0)  # Last entry corresponds to exterior region.
    solution_types.append(["exterior"])
    residual = []

    for i, (phi_b, VEV_L, VEV_R, delta_L, delta_R, sol_type_L, sol_type_R) \
        in enumerate(zip(boundary_values, field_VEVs[:-1], field_VEVs[1:],
                         deltas[:-1], deltas[1:], solution_types[:-1],
                         solution_types[1:])):

        x_L = phi_b/VEV_L
        x_R = phi_b/VEV_R

        # Values for boundaries to left and right.
        if i == 0:
            phi_L = VEV_L
        else:
            phi_L = boundary_values[i-1]

        if i == len(boundary_values) - 1:
            phi_R = VEV_R
        else:
            phi_R = boundary_values[i+1]

        # Determine signs of gradient on left side.
        if sol_type_L[0] == "exterior":
            sig_L = np.sign(phi_b - VEV_L)

        elif sol_type_L[2] is True:
            # This means curve is monotonic.
            sig_L = np.sign(phi_b - phi_L)

        else:
            # Must be have extrema in region.
            if sol_type_L[1] == "maximum":
                sig_L = -1
            else:
                sig_L = +1

        # Determine signs of gradient on right side.
        if sol_type_R[0] == "exterior":
            sig_R = np.sign(VEV_R - phi_b)

        elif sol_type_R[2] is True:
            # This means curve is monotonic.
            sig_R = np.sign(phi_R - phi_b)

        else:
            # Must be have extrema in region.
            if sol_type_R[1] == "maximum":
                sig_R = +1
            else:
                sig_R = -1

        # Record residual. See Eq. (50).
        residual.append(sig_L*np.sqrt(max((x_L - 1)**2 - x_L*delta_L, 0.0)) -
                        sig_R*np.sqrt(max((x_R - 1)**2 - x_R*delta_R, 0.0)))

    return np.array(residual)

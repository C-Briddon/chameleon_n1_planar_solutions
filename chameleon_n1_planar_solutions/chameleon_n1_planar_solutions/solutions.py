#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:58:49 2026

@author: chad-briddon

Library of solutions to the 1D planar chameleon equation of motion.

Note VEV stands for vacuum expectation value.

All numpy arrays assume values moving from left --> right.

"""

import warnings
import numpy as np

from astropy import units, constants
from scipy.optimize import least_squares


from chameleon_n1_planar_solutions.elliptic_functions import (
    K, sn, ns, sn_Inverse, ns_Inverse, Sn, Ns, Sc, Sn_Inverse, Ns_Inverse,
    dc_minus_1, nc_minus_1)

from chameleon_n1_planar_solutions.boundary_functions import (
    solve_for_parameter, residual_boundary_constraint,
    w_of_y, y_of_w, u_of_y, y_of_u, extremum_test, G)

from chameleon_n1_planar_solutions.chameleon_identities import (
    calc_VEV, calc_Compton_WL)


' -------------------------------------------------------------------------- '
' ------------------- Define solutions for each scenario ------------------- '
' -------------------------------------------------------------------------- '


def exterior(field_VEV, compton_WL, z_boundary, field_boundary):
    '''
    Generate solution inside exterior regions given field value at boundary.
    See Eq. (10).

    Parameters
    ----------
    field_VEV : float
        VEV corresponding to region.
    compton_WL : float
        Compton wavelength corresponding to field VEV.
    z_boundary : float
        Location of the boundary.
    field_boundary : float
        Value of field at boundary.


    Returns
    -------
    function
        Function that return field value in exterior given position z as
        argument.

    '''

    x_b = np.sqrt(field_boundary/field_VEV)
    G_b, up = G(x_b), x_b > 1  # G grows away from x = 1 on both sides

    def invert(zi):
        if x_b == 1:
            return 1.0
        target = G_b - abs(zi - z_boundary)/(2*compton_WL)
        a, b = min(1, x_b), max(1, x_b)
        while abs(b - a) > 1e-14*max(1, b):
            c = (a + b)/2
            if (G(c) > target) == up:
                b = c
            else:
                a = c
        return (a + b)/2

    def f(z):
        if np.ndim(z) == 0:
            return field_VEV*invert(z)**2
        return field_VEV*np.array([invert(zi) for zi in z])**2

    return f


def interior(field_VEV, compton_WL, z_boundary_L, z_boundary_R,
             field_boundary_L, field_boundary_R, gamma):
    '''
    Generate solution inside interior regions given field values at both
    boundaries and the curve parameter.
    See Eqs. (15), (25), (38), and (40).

    Parameters
    ----------
    field_VEV : float
        VEV corresponding to region.
    compton_WL : float
        Compton wavelength corresponding to field VEV.
    z_boundary_L : float
        Location of the left boundary.
    z_boundary_R : float
        Location of the right boundary.
    field_boundary_L : float
        Value of field at left boundary.
    field_boundary_R : float
        Value of field at right boundary.
    gamma : float
        Placeholder parameter for the curve parameter. If extrema case then
        'gamma = k' and if VEV case '|C| = (1 - gamma)/gamma'. Note in either
        case gamma in range (0, 1).

    Returns
    -------
    function
        Function that return field value in interior given position z as
        argument.

    '''

    L_i = z_boundary_R - z_boundary_L
    z_c = (z_boundary_R + z_boundary_L)/2  # Centre coordinate.

    is_extremum = extremum_test(field_boundary_L, field_boundary_R,
                                field_VEV, L_RS=L_i/(2*compton_WL))

    if is_extremum:

        if max(field_boundary_L, field_boundary_R) < field_VEV:
            ' ---- Scenario : extrema (maximum) ---- '

            k2 = gamma**2
            L_c = 2*compton_WL*gamma**1.5

            x_L = field_boundary_L/(field_VEV*gamma)
            x_R = field_boundary_R/(field_VEV*gamma)

            # Calculate x_c using Eq. (17).
            Sn_K = Sn(K(k2), k2)
            x_c = sn(Sn_Inverse(Sn_K - (L_i/L_c), k2), k2)**2

            # Calculate z_0 using Eq. (16).
            if min(x_L, x_R) > x_c:
                # Maximum occurs inside region.
                z_0 = (Sn(sn_Inverse(np.sqrt(x_L), k2), k2) -
                       Sn(sn_Inverse(np.sqrt(x_R), k2), k2))/2
            else:
                # Maximum occurs outside region (is fictitious).
                z_0 = Sn_K - (Sn(sn_Inverse(np.sqrt(x_L), k2), k2) +
                              Sn(sn_Inverse(np.sqrt(x_R), k2), k2))/2

                # Get correct sign.
                if x_L < x_R:
                    z_0 *= -1
                else:
                    pass

            # Give field profile as a callable function of Eq. (15).
            def f(z):
                '''
                Returned function for field as function of z in interior region
                (extrema case with maxima).

                Parameters
                ----------
                z : float or numpy.array
                    Position value(s).

                Returns
                -------
                float
                    Field value at position z.

                '''

                zprime = z - z_c

                return field_VEV*gamma*sn(Sn_Inverse(
                    Sn_K - abs((zprime/L_c) + z_0), k2), k2)**2

        elif min(field_boundary_L, field_boundary_R) > field_VEV:
            ' ---- Scenario : extrema (minimum) ---- '

            g2 = 1/gamma**2  # use g := 1/k.
            L_c = 2*compton_WL*np.sqrt(gamma)

            x_L = field_boundary_L/(field_VEV*gamma)
            x_R = field_boundary_R/(field_VEV*gamma)

            # Calculate x_c using Eq. (27).
            Ns_K = Ns(K(g2), g2)
            x_c = ns(Ns_Inverse(Ns_K - (L_i/L_c), g2), g2)**2

            # Calculate z_0 using Eq. (26).
            if max(x_L, x_R) < x_c:
                # Minimum occurs inside region.
                z_0 = (Ns(ns_Inverse(np.sqrt(x_L), g2), g2) -
                       Ns(ns_Inverse(np.sqrt(x_R), g2), g2))/2
            else:
                # Minimum occurs outside region (is fictitious).
                z_0 = Ns_K - (Ns(ns_Inverse(np.sqrt(x_L), g2), g2) +
                              Ns(ns_Inverse(np.sqrt(x_R), g2), g2))/2

                # Get correct sign.
                if x_L > x_R:
                    z_0 *= -1
                else:
                    pass

            # Give field profile as a callable function of Eq. (25).
            def f(z):
                '''
                Returned function for field as function of z in interior region
                (extrema case with minima).

                Parameters
                ----------
                z : float or numpy.array
                    Position value(s).

                Returns
                -------
                float
                    Field value at position z.

                '''

                zprime = z - z_c

                return field_VEV*gamma*ns(Ns_Inverse(
                    Ns_K - abs((zprime/L_c) + z_0), g2), g2)**2

    else:
        ' ---- Scenario : passes VEV ---- '

        C_abs = abs(gamma)
        C2 = C_abs**2

        y_L = field_boundary_L/field_VEV
        y_R = field_boundary_R/field_VEV

        if C_abs >= 1:
            ' ---- Scenario : passes VEV (C^2 >= 1) ---- '

            mu2 = 1 - 1/C2

            u_L = u_of_y(y_L, mu2)
            u_R = u_of_y(y_R, mu2)

            u_min = min(u_L, u_R)
            u_max = max(u_L, u_R)

            sig = np.sign(y_R - y_L)

            # Calculate z_0 by evaluating Eq. (38) at left boundary.
            def residual_int(u):
                return (u + 2*Sc(u, mu2, mc=1/C2))/C_abs \
                    + 2*C_abs*dc_minus_1(u, mu2, mc=1/C2)

            z_star = z_boundary_L - sig*compton_WL*residual_int(u_L)

            # Give field profile as a callable function using bisection method.
            def f(z):
                '''
                Returned function for field as function of z in interior region
                (passes VEV case with C**2 => 1).

                Uses bisection method to invert non-invertable equation.

                Parameters
                ----------
                z : float or numpy.array
                    Position value(s).

                Returns
                -------
                float
                    Field value at position z.

                '''

                Z = sig*(z - z_star)/compton_WL

                if isinstance(Z, float):

                    a, b = u_min, u_max

                    while abs(b-a) > 1e-14:

                        c = (a + b)/2

                        if residual_int(c) > Z:
                            b = c
                        else:
                            a = c

                    u = (a + b)/2

                else:

                    inv = []
                    for Zi in Z:

                        a, b = u_min, u_max

                        while abs(b-a) > 1e-14:

                            c = (a + b)/2

                            if residual_int(c) > Zi:
                                b = c
                            else:
                                a = c

                        inv.append((a + b)/2)

                    u = np.array(inv)

                # Convert from u to phi.
                field = field_VEV*y_of_u(u, mu2)

                return field

        else:
            ' ---- Scenario : passes VEV (C^2 < 1) ---- '

            nu2 = 1 - C2

            w_L = w_of_y(y_L, C_abs, nu2)
            w_R = w_of_y(y_R, C_abs, nu2)

            w_min = min(w_L, w_R)
            w_max = max(w_L, w_R)

            sig = np.sign(y_R - y_L)

            # Calculate z_0 by evaluating Eq. (40) at left boundary.
            def residual_int(w):
                return w + 2*C2*Sc(w, nu2, mc=C2) + 2*C_abs*nc_minus_1(w, nu2)

            z_star = z_boundary_L - sig*compton_WL*residual_int(w_L)

            # Give field profile as a callable function using bisection method.
            def f(z):
                '''
                Returned function for field as function of z in interior region
                (passes VEV case with C**2 < 1).

                Uses bisection method to invert non-invertable equation.

                Parameters
                ----------
                z : float or numpy.array
                    Position value(s).

                Returns
                -------
                float
                    Field value at position z.

                '''

                Z = sig*(z - z_star)/compton_WL

                if isinstance(Z, float):

                    a, b = w_min, w_max

                    while abs(b-a) > 1e-14:

                        c = (a + b)/2

                        if residual_int(c) > Z:
                            b = c
                        else:
                            a = c

                    w = (a + b)/2

                else:

                    inv = []
                    for Zi in Z:

                        a, b = w_min, w_max

                        while abs(b-a) > 1e-14:

                            c = (a + b)/2

                            if residual_int(c) > Zi:
                                b = c
                            else:
                                a = c

                        inv.append((a + b)/2)

                    w = np.array(inv)

                # Convert from w to phi.
                field = field_VEV*y_of_w(w, C_abs, nu2)

                return field

    return f


' --------------------------------------------------------------------------- '
' ------------------ Generate solutions for N-layered case ------------------ '
' --------------------------------------------------------------------------- '


def planar_solution(Lambda, M, densities, boundary_positions,
                    boundary_guess=None, density_NonEVUnits=None,
                    length_NonEVUnits=None, use_boundary_guess=False):
    '''
    Compute field boundary values and use then to generate a function that
    returns the field profile of a general N-layered system.

    Parameters
    ----------
    Lambda : float
        Mass scale of field potential (defined in eV).
    M : float
        Coupling mass scale (defined in eV).
    densities : numpy.array
        Background density values.
    boundary_positions : numpy.array
        Array of positions of the boundaries between neighbouring regions.
    boundary_guess : numpy.array, optional
        Initial guess for the boundary values. If None then harmonic mean field
        value corresponding to each boundary is used. The default is None.
    density_NonEVUnits : astropy.units, optional
        Define the phyical units of density used. If None then provided density
        values are taken to be in units of eV. The default is None.
    length_NonEVUnits : astropy.units, optional
        Define the phyical units of length used. If None then provided lengths
        values are taken to be in units of eV. The default is None.
    use_boundary_guess : bool, optional
        If True then the solution is generated using the boundary values
        used as an initial guess. The default is False.

    Returns
    -------
    f : function
        A function that takes positional arguments of z and return the
        corresponding field value.
    boundary_values : numpy.array
        Contains boundary values used to construct f.
    curve_parameters : list of tuples
        Each entry of list corresponds to an interior region. Tuple entries
        give solution solution type (extrema or VEV) and the second gives the
        curve parameter value. In extrema case 'curve_parameters' is 'k' and
        in the VEV case is '1/(1 + |C|)'. In either case the range of
        'curve_parameter' is (0, 1).

    '''

    curve_parameters = []

    # Convert density to natural units.
    if density_NonEVUnits:
        kg_per_m3_to_eV4 = (constants.hbar.value**3)*(constants.c.value**5)/(
            constants.e.value**4)

        densities_eV = densities *\
            density_NonEVUnits.to(units.kg*units.m**(-3))*kg_per_m3_to_eV4
    else:
        densities_eV = densities

    # Get relavent field quantities in each region.
    field_VEVs = calc_VEV(Lambda=Lambda, M=M, density=densities_eV)
    compton_WLs = calc_Compton_WL(Lambda=Lambda, M=M, density=densities_eV,
                                  length_NonEVUnits=length_NonEVUnits)

    if len(densities_eV) > 2:
        L = np.array([(z_R - z_L) for z_L, z_R in zip(boundary_positions[:-1],
                                                      boundary_positions[1:])])

        L_RSs = L/(2*compton_WLs[1:-1])

        # If no guess is provided, use average of neighbouring VEVs.
        if boundary_guess is None:
            boundary_guess = 2*np.sqrt(M*Lambda**5)/(
                np.sqrt(densities_eV[1:]) + np.sqrt(densities_eV[:-1]))

        # Compute boundary values (or use initial guess).
        if use_boundary_guess is True:
            boundary_values = boundary_guess

        else:
            field_min = min(field_VEVs)
            field_max = max(field_VEVs)

            res = least_squares(
                fun=residual_boundary_constraint, x0=boundary_guess,
                bounds=(field_min, field_max), args=(field_VEVs, L_RSs))

            if not res.success or np.max(np.abs(res.fun)) > 1e-8:
                warnings.warn(
                    f"Boundary solve may not have converged: {res.message} "
                    f"(max |residual| = {np.max(np.abs(res.fun)):.1e})")

            boundary_values = res.x

    else:
        boundary_values = [2*np.sqrt(M*Lambda**5) /
                           (np.sqrt(densities_eV[0])
                            + np.sqrt(densities_eV[1]))]

    # Construct profiles : start with exterior region.
    profiles = [exterior(field_VEVs[0], compton_WLs[0], boundary_positions[0],
                         boundary_values[0])]

    # Construct profiles : all interior layers.
    for z_L, z_R, phi_L, phi_R, field_VEV, compton_WL in zip(
            boundary_positions[:-1], boundary_positions[1:],
            boundary_values[:-1], boundary_values[1:],
            field_VEVs[1:], compton_WLs[1:]):

        L_rs = (z_R - z_L)/(2*compton_WL)
        is_extremum = extremum_test(phi_L, phi_R, field_VEV, L_rs)

        sol_type, para = solve_for_parameter(data=(L_rs, field_VEV,
                                                   phi_L, phi_R, is_extremum))

        if sol_type[0] == "VEV":
            gamma = (1 - para)/para
        else:
            gamma = para

        profiles.append(
            interior(field_VEV, compton_WL, z_L, z_R, phi_L, phi_R, gamma))

        curve_parameters.append((sol_type[0], gamma))

    # Construct profiles : end with exterior region.
    profiles.append(exterior(field_VEVs[-1], compton_WLs[-1],
                             boundary_positions[-1], boundary_values[-1]))

    # Combine profiles into a single function.
    def f(z):
        '''
        Function that returns field as a function of position, z.

        Parameters
        ----------
        z : float or numpy.array
            Position value(s).

        Returns
        -------
        field : numpy.array
            Field values.

        '''

        if isinstance(z, float):

            # Check for right exterior region.
            if z > boundary_positions[-1]:
                field = profiles[-1](z)
            else:
                # Otherwise.
                for j, zb in enumerate(boundary_positions):
                    if z <= zb:
                        field = profiles[j](z)
                        break

        else:

            field = np.zeros(len(z))

            for i, zi in enumerate(z):

                # Check for right exterior region.
                if zi > boundary_positions[-1]:
                    field[i] = profiles[-1](zi)
                else:
                    # Otherwise.
                    for j, zb in enumerate(boundary_positions):
                        if zi <= zb:
                            field[i] = profiles[j](zi)
                            break

        return field

    return f, boundary_values, curve_parameters

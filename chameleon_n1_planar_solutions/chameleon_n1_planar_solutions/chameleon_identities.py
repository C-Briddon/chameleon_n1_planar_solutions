#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:58:49 2026

@author: chad-briddon

Library of functions for quantities related to the n=1 chameleon field.

In 'Exact analytic solutions for the n=1 chameleon field in planar systems with
an arbitrary number of uniform layers' see Eqs (3) and (5).

"""

import numpy as np

from astropy import units, constants


def calc_VEV(Lambda, M, density):
    '''
    Calculate the VEV (vacuum expectation value) of the n=1 chameleon model.

    Parameters
    ----------
    Lambda : float
        Mass scale of field potential (defined in eV).
    M : float
        Coupling mass scale (defined in eV).
    density : numpy.array
        Background density values (defined in eV).

    Returns
    -------
    VEV : float
        Field VEV corresponding to given density (defined in eV).

    '''

    C = M*Lambda**5
    VEV = np.sqrt(C/density)

    return VEV


def calc_Compton_WL(Lambda, M, density, length_NonEVUnits=None):
    '''
    Calculate the n=1 chameleon field's Compton wavelength in a region of
    uniform density.

    Parameters
    ----------
    Lambda : float
        Mass scale of field potential (defined in eV).
    M : float
        Coupling mass scale (defined in eV).
    density : numpy.array
        Background density values (defined in eV).
    length_NonEVUnits : astropy.units, optional
        If None the return Compton wavelength will be in units of eV.
        If desire other units, input them using this argument.
        The default is None.

    Returns
    -------
    comp_wl : float
        Compton wavelength of the n=1 chameleon field, given in units
        determined by 'length_NonEVUnits'.

    '''

    comp_wl = (Lambda**1.25)*((M/density)**0.75)/np.sqrt(2)

    # Change units if needed.
    if length_NonEVUnits:
        _per_m_to_eV = constants.c.value*constants.hbar.value/constants.e.value
        comp_wl /= length_NonEVUnits.to(units.m)/_per_m_to_eV

    return comp_wl

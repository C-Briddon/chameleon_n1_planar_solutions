#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:27:29 2026

@author: chad-briddon

Check exterior solutions.

"""

import numpy as np
import matplotlib.pyplot as plt

import chameleon_n1_planar_solutions.solutions as sol
import chameleon_n1_planar_solutions.chameleon_identities as ci


option = 1


' -------------------------------------------------------------------------- '

plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)


' -------------------------------------------------------------------------- '
' ----------------------------- Set parameters ----------------------------- '
' -------------------------------------------------------------------------- '

if option == 1:
    L = 1                                   # eV
    density = 1                             # eV
    Lambda = 1e-1                           # eV
    M = 1e0                                 # eV

    field_L = 0.2                           # eV

elif option == 2:
    L = 2                                   # eV
    density = 1                             # eV
    Lambda = 1e-1                           # eV
    M = 2e1                                 # eV

    field_L = 0.005                         # eV


z = np.linspace(0, L, 1000)                 # eV


' -------------------------------------------------------------------------- '
' ----------------- Use functions to get analytic solution ----------------- '
' -------------------------------------------------------------------------- '

field_VEV = ci.calc_VEV(Lambda, M, density)
compton_WL = ci.calc_Compton_WL(Lambda, M, density)

# Construct function for profile.
f = sol.exterior(field_VEV, compton_WL, z_boundary=0.0, field_boundary=field_L)

field_anal = f(z)


' --------------------------------------------------------------------------- '
' ----------------------- Plot and compare two curves ----------------------- '
' --------------------------------------------------------------------------- '

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel(r"$\hat{\phi}$")
ax.set_xlabel("z")
ax.set_xlim([0, L])

ax.axhline(field_VEV, linestyle="dashdot", color="r", label="VEV")

ax.plot(z, field_anal, label="Analytic", linestyle="solid")
ax.plot([0], [field_L], "r.")

ax.legend()
plt.show()

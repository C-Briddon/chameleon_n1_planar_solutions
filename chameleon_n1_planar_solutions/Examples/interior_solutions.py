#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:00:34 2026

@author: chad-briddon

Check interior solutions.

"""

import numpy as np
import matplotlib.pyplot as plt

import chameleon_n1_planar_solutions.solutions as sol
import chameleon_n1_planar_solutions.boundary_functions as bf
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

    field_L = 0.1                           # eV
    field_R = 0.2                           # eV

elif option == 2:
    L = 1                                   # eV
    density = 1                             # eV
    Lambda = 1e-1                           # eV
    M = 2e1                                 # eV

    field_L = 0.010                         # eV
    field_R = 0.005                         # eV

elif option == 3:
    L = 1                                   # eV
    density = 1                             # eV
    Lambda = 1e-1                           # eV
    M = 2e1                                 # eV

    field_L = 0.014                         # eV
    field_R = 0.010                         # eV


z = np.linspace(-L/2, L/2, 1000)            # eV


' -------------------------------------------------------------------------- '
' ----------------- Use functions to get analytic solution ----------------- '
' -------------------------------------------------------------------------- '

field_VEV = ci.calc_VEV(Lambda, M, density)
compton_WL = ci.calc_Compton_WL(Lambda, M, density)
L_rs = 0.5*L/compton_WL

# Get curve parameter.
is_extremum = bf.extremum_test(field_L, field_R, field_VEV, L_rs)
solution_type, curve_parameter = bf.solve_for_parameter(
    (L_rs, field_VEV, field_L, field_R, is_extremum))

if solution_type[0] == "VEV":
    curve_parameter = (1/curve_parameter) - 1

print()
print("Solution type =", solution_type)
print()
print("Curve parameter =", curve_parameter)

# Construct function for profile.
f = sol.interior(field_VEV, compton_WL,
                 z_boundary_L=-L/2,
                 z_boundary_R=+L/2,
                 field_boundary_L=field_L,
                 field_boundary_R=field_R,
                 gamma=curve_parameter)

field_anal = f(z)


' --------------------------------------------------------------------------- '
' ----------------------- Plot and compare two curves ----------------------- '
' --------------------------------------------------------------------------- '

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel(r"$\hat{\phi}$")
ax.set_xlabel("z")
ax.set_xlim([-L/2, L/2])

ax.axhline(field_VEV, linestyle="dashdot", color="r", label="VEV")

ax.plot(z, field_anal, label="Analytic", linestyle="solid")
ax.plot([-L/2], [field_L], "r.")
ax.plot([+L/2], [field_R], "r.")

ax.legend()
plt.show()

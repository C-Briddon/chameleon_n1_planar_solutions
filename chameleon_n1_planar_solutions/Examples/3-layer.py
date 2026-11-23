#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:13:30 2026

@author: chad-briddon

Example of 3-layer solution. Compare analytic and numeric solutions.

*This code ran using SELCIE v2.0.0.

"""

import numpy as np
import matplotlib.pyplot as plt

from SELCIE.Solvers import Chameleon
from SELCIE.Mesh import MeshingTools_1D
from SELCIE.Misc import calc_field_min, alpha_calculator_chameleon

from chameleon_n1_planar_solutions.solutions import planar_solution


option = 1

' -------------------------------------------------------------------------- '

plt.rc('axes', labelsize=15)
plt.rc('legend', fontsize=15)


' -------------------------------------------------------------------------- '
' ----------------------------- Set parameters ----------------------------- '
' -------------------------------------------------------------------------- '

if option == 1:
    P_R = 1e4                               # eV
    P_L = 1e5                               # eV
    P_V = 1                                 # eV

    Lambda = 1e-1                           # eV
    M = 1e4                                 # eV

elif option == 2:
    P_R = 1e4                               # eV
    P_L = 1e5                               # eV
    P_V = 2e4                               # eV

    Lambda = 1e-1                           # eV
    M = 2e4                                 # eV

elif option == 3:
    P_R = 1.5                               # eV
    P_L = 1                                 # eV
    P_V = 2                                 # eV

    Lambda = 1e-1                           # eV
    M = 5                                   # eV


field_scale = calc_field_min(n=1, M=M, Lam=Lambda, p0=1)
alpha = alpha_calculator_chameleon(n=1, M=M, Lam=Lambda, p0=1, L=1)

z = np.linspace(-1.6, 1.6, 1000)            # eV


' -------------------------------------------------------------------------- '
' ----------------- Use functions to get analytic solution ----------------- '
' -------------------------------------------------------------------------- '

f, boundary_values, curve_parameters = planar_solution(
    Lambda, M, densities=np.array([P_L, P_V, P_R]),
    boundary_positions=np.array([-0.5, +0.5]), density_NonEVUnits=None,
    length_NonEVUnits=None)

field_anal = f(z)


' -------------------------------------------------------------------------- '
' --------------------- Solve numerically using SELCIE --------------------- '
' -------------------------------------------------------------------------- '


def create_mesh_parellel_plates(show_mesh=True):

    filename = "parellel_plates"

    # Make mesh.
    MT = MeshingTools_1D(display_messages=False)

    plate_L = MT.create_1D_line(-2, -0.5)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    plate_R = MT.create_1D_line(0.5, 5)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    MT.create_1D_line(-0.51, 0.51,
                      embed=plate_L+plate_R)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.05,
                        DistMax=0.2)

    MT.generate_mesh(filename, show_mesh)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def make_density_func(p):

    def f(x):
        return p

    return f


mesh_name = create_mesh_parellel_plates(show_mesh=False)

left = make_density_func(P_L)
right = make_density_func(P_R)
vacuum = make_density_func(P_V)

s = Chameleon(alpha, n=1, mesh_name=mesh_name, dimension=1,
              symmetry="translation symmetry")

s.solve_static(density_profiles=[left, right, vacuum], tol_du=1e-10)

field_numc = field_scale*np.array([s.field(zi, 0) for zi in z])


' --------------------------------------------------------------------------- '
' ----------------------- Plot and compare two curves ----------------------- '
' --------------------------------------------------------------------------- '

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title("Field profile (numeric & analytic)")
ax1.set_ylabel(r"$\hat{\phi}$")
ax1.set_xlabel("z")
ax1.set_xlim([-1.6, 1.6])

ax1.axvline(-1/2, linestyle="dashdot", color="k")
ax1.axvline(+1/2, linestyle="dashdot", color="k")

ax1.axhline(calc_field_min(n=1, M=M, Lam=Lambda, p0=P_L),
            linestyle="dashdot", color="r", label=r"$VEV_L$")

ax1.axhline(calc_field_min(n=1, M=M, Lam=Lambda, p0=P_R),
            linestyle="dashdot", color="c", label=r"$VEV_R$")

ax1.plot(z, field_numc, label="Numeric", linestyle="solid")
ax1.plot(z, field_anal, label="Analytic", linestyle="dashed")

ax1.legend()
plt.show()

# Plot difference.
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title("Relative error (numeric vs analytic)")
ax2.set_ylabel(r"$|\Delta \hat{\phi}|$")
ax2.set_xlabel("z")
ax2.set_xlim([-1.6, 1.6])
ax2.set_yscale("log")

ax2.plot(z, abs(field_numc - field_anal))

plt.show()

'''
Note the relative error scales with mesh refinement. Mesh length can also lead
to an error if field does not have enough space to reach its VEV.
'''

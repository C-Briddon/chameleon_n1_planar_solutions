#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:00:10 2026

@author: chad-briddon

Example of 4-layer solution. Compare analytic and numeric solutions.

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
    P_1 = 1e3                               # eV
    P_2 = 1                                 # eV
    P_3 = 1e5                               # eV
    P_4 = 10                                # eV

    Lambda = 1e-1                           # eV
    M = 1e4                                 # eV

elif option == 2:
    P_1 = 1e3                               # eV
    P_2 = 1                                 # eV
    P_3 = 7e4                               # eV
    P_4 = 100                               # eV

    Lambda = 1e-1                           # eV
    M = 1e5                                 # eV


field_scale = calc_field_min(n=1, M=M, Lam=Lambda, p0=1)
alpha = alpha_calculator_chameleon(n=1, M=M, Lam=Lambda, p0=1, L=1)

z = np.linspace(-6, 6, 1000)                # eV


' -------------------------------------------------------------------------- '
' ----------------- Use functions to get analytic solution ----------------- '
' -------------------------------------------------------------------------- '

f, boundary_values, curve_parameters = planar_solution(
    Lambda, M, densities=np.array([P_1, P_2, P_3, P_4]),
    boundary_positions=np.array([0.0, 1.0, 1.2]),
    density_NonEVUnits=None, length_NonEVUnits=None)

field_anal = f(z)


' -------------------------------------------------------------------------- '
' --------------------- Solve numerically using SELCIE --------------------- '
' -------------------------------------------------------------------------- '


def create_mesh_parellel_plates(show_mesh=True):

    filename = "Casimir_plates"

    # Make mesh.
    MT = MeshingTools_1D(display_messages=False)

    plate_1 = MT.create_1D_line(-6, 0.0)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    plate_2 = MT.create_1D_line(-0.1, 1.0, embed=plate_1)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    plate_3 = MT.create_1D_line(0.9, 1.2, embed=plate_1+plate_2)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    MT.create_1D_line(1.1, 50, embed=plate_1+plate_2+plate_3)
    MT.create_subdomain(CellSizeMin=1e-5, CellSizeMax=0.001, DistMin=0.0,
                        DistMax=0.2)

    MT.generate_mesh(filename, show_mesh)
    MT.msh_2_xdmf(filename, auto_override=True)

    return filename


def make_density_func(p):

    def f(x):
        return p

    return f


mesh_name = create_mesh_parellel_plates(show_mesh=False)

s = Chameleon(alpha, n=1, mesh_name=mesh_name, dimension=1,
              symmetry="translation symmetry")

s.solve_static(density_profiles=[make_density_func(P_1),
                                 make_density_func(P_2),
                                 make_density_func(P_3),
                                 make_density_func(P_4)], tol_du=1e-9)

field_numc = field_scale*np.array([s.field(zi, 0) for zi in z])


' --------------------------------------------------------------------------- '
' ----------------------- Plot and compare two curves ----------------------- '
' --------------------------------------------------------------------------- '

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title("Field profile (numeric & analytic)")
ax1.set_ylabel(r"$\hat{\phi}$")
ax1.set_xlabel("z")
ax1.set_xlim([-6, 6])

ax1.axvline(0.0, linestyle="dashdot", color="k")
ax1.axvline(1.0, linestyle="dashdot", color="k")
ax1.axvline(1.2, linestyle="dashdot", color="k")
ax1.axvline(6.0, linestyle="dashdot", color="k")

ax1.axhline(calc_field_min(n=1, M=M, Lam=Lambda, p0=P_1),
            linestyle="dashdot", color="r", label=r"$VEV_1$")

ax1.axhline(calc_field_min(n=1, M=M, Lam=Lambda, p0=P_3),
            linestyle="dashdot", color="c", label=r"$VEV_3$")

ax1.plot(z, field_numc, label="Numeric", linestyle="solid")
ax1.plot(z, field_anal, label="Analytic", linestyle="dashed")

ax1.legend()
plt.show()

# Plot difference.
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title("Relative error (numeric vs analytic)")
ax2.set_ylabel(r"$|\Delta \hat{\phi}|$")
ax2.set_xlabel("z")
ax2.set_xlim([-6, 6])
ax2.set_yscale("log")

ax2.plot(z, abs(field_numc - field_anal))

plt.show()

'''
Note the relative error scales with mesh refinement. Mesh length can also lead
to an error if field does not have enough space to reach its VEV.
'''

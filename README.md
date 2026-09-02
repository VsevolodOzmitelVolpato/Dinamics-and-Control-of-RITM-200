# Dinamics-and-Control-of-RITM-200
This is my Bs thesis project, this Python repository features a coupled numerical model simulating the dynamic behavior of the RITM-200 SMR nuclear reactor. It integrates point neutron kinetics, core thermal-hydraulics, and a 1D steam generator to analyze operational transients like load following and control rod insertion and other features.
# ⚛️ RITM-200 Dynamics and Control Simulation

## Overview
This is my Bs thesis project and it presents a coupled numerical model developed in Python to analyze the dynamic behavior of the **RITM-200 Small Modular Reactor (SMR)**. The main objective is to simulate and understand how the reactor responds to operational transients and perturbations, highlighting the interdependence of its neutronics, thermal-hydraulics, and thermodynamic systems.

## Modeled Subsystems
To capture the real-world behavior of the reactor, the model integrates several distinct but interacting physical subsystems:

- **Neutron Kinetics** — The reactor's neutron population is modeled using point reactor kinetics with six groups of delayed neutrons. It accounts for reactivity feedback mechanisms, primarily the Doppler effect (fuel temperature), moderator density variations (coolant temperature), and the external control provided by the control rods.
- **Core Thermal-Hydraulics** — The core's thermal dynamics are represented using a lumped-parameter approach. To accurately capture internal temperature gradients, the fuel element is radially discretized into three concentric nodes, alongside representations of the cladding and the primary coolant.
- **Steam Generator (SG)** — The heat exchanger is modeled as a one-dimensional, counter-current, cassette-type system of concentric tubes. It tracks heat transfer from the primary to the secondary circuit and dynamically handles the secondary fluid's phase changes (subcooled liquid, two-phase boiling, and superheated steam).
- **Secondary Pressure Dynamics** — A phenomenological model simulates how the pressure in the secondary circuit fluctuates in response to the steam flow rate drawn by the turbine.

## Numerical Methods and Tools
The simulation is entirely implemented in Python, using standard scientific libraries to handle the complex system dynamics:

| Component | Method / Tool |
|---|---|
| Matrix operations & solvers | `NumPy`, `SciPy` |
| Water/steam properties | `pyXSteam` |
| Core integration | `LSODA` (handles the "stiff" nature of the neutronics/thermal ODEs) |
| Steam generator scheme | Unconditionally stable **implicit Euler** |
| System coupling | *Hot leg* / *cold leg* temperature exchange between core and SG, with reactivity updated at every time step |

## Simulated Operational Scenarios
Two main transient scenarios were simulated to validate the model and observe the reactor's stability:

### 1. Load Following (Downstream Power Control)
**What was done:** A +10% step increase in the secondary circuit's mass flow rate was introduced to simulate an increased energy demand from the turbine.

**Observed behavior:** The higher steam extraction lowers the secondary pressure, which improves heat transfer and cools the primary coolant further. When this colder, denser water reaches the core, it acts as a better moderator, introducing positive reactivity. The reactor automatically increases its thermal power to match the new turbine demand — demonstrating the typical **load-following** capability of PWR-type reactors.

### 2. Control Rod Insertion (Upstream Power Control)
**What was done:** The simulation tested a direct insertion of control rods into the core while keeping the secondary flow rate constant.

**Observed behavior:** The rods immediately absorb neutrons, causing a sharp drop in neutron density and inhibiting the fission chain reaction. This leads to a rapid decrease in fuel temperature and less heat transferred to the primary coolant. Although the cooling of the water slightly increases its moderating capacity (causing a minor rebound in neutron density), the system safely stabilizes at a new, lower power equilibrium.

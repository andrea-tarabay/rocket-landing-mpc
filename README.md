# Rocket Landing with Model Predictive Control

**EPFL — ME-425 Model Predictive Control**

This project develops and compares several MPC strategies for landing a thrust-vector-controlled rocket while respecting state and actuator constraints.

## Methods

- Linear MPC for velocity and attitude regulation
- Cascaded PI + MPC position control
- Offset-free tracking under mass mismatch
- Robust Tube MPC for disturbed landing
- Nonlinear MPC using the full 12-state rocket dynamics

The NMPC implementation uses **RK4 discretization** and is solved with **CasADi/IPOPT**.

## Results

Both Robust MPC and NMPC successfully land the rocket.

- Position convergence in approximately **2–3 s**
- Vertical landing target reached in approximately **3–4 s**
- Robust Tube MPC provides more conservative behavior under disturbances
- NMPC captures nonlinear coupling more directly

## Repository Structure

```text
Deliverable_3_1/   Linear MPC regulation
Deliverable_3_2/   Reference tracking
Deliverable_3_3/   Cascaded PI + MPC position control
Deliverable_4_1/   Nonlinear rocket simulation
Deliverable_5_1/   Offset-free tracking
Deliverable_5_2/   Changing-mass experiments
Deliverable_6_1&2/ Robust Tube MPC
Deliverable_7_1/   Nonlinear MPC
PIControl/         Outer-loop PI controller
src/               Rocket model and simulation utilities
```

## Team

- Andrea Tarabay
- Masa Duric
- Aayushi Kedar Barve

## Report

📄 [Full Project Report](MPC_Project_Report.pdf)

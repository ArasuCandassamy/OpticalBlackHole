# Optical Black Hole

This repository contains code and resources related to a project for PHY_51171_EP - Numerical Physics at Ecole Polytechnique. The project focuses on simulating and analyzing the properties of optical black holes using numerical methods.

## Project Structure
- `design/`: Contains Jupyter notebooks and scripts for designing and simulating the optical black hole.
  - `raytracing.ipynb`: Notebook for ray tracing simulations around the optical black hole.
  - `FDFD.ipynb`: Notebook implementing the Finite-Difference Frequency-Domain (FDFD) method for simulating wave propagation.
  - `FDFD_convergence.ipynb`: Notebook analyzing the convergence of the FDFD method.

- `production/`: Contains scripts and resources for the production phase of the optical black hole project.
    - `FDFD_simulation.py`: Python script for running FDFD simulations.
    To run the simulation, use the command:
    ```bash
    python FDFD_simulation.py
    ```
    Then one can choose parameters for the simulation in the terminal prompt.
    - `plotting.py`: Python script for generating plots from simulation data.


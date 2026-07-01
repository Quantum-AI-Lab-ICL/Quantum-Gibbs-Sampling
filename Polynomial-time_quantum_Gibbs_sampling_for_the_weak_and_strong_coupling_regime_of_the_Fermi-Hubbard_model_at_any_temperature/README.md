# Quantum-Gibbs-Sampling

[![DOI](https://zenodo.org/badge/909009541.svg)](https://doi.org/10.5281/zenodo.17390409)

This repository contains the code for exact numerical simulations of fermionic quantum Gibbs sampling from the paper *Polynomial-time quantum Gibbs sampling for the weak and strong coupling regime of the Fermi-Hubbard model at any temperature*; authored by Štěpán Šmíd, Richard Meister, Mario Berta, and Roberto Bondesan, and available at https://www.nature.com/articles/s41467-025-65765-1.

The structure of the Python code is very simple, and it only needs numpy, scipy, and openfermion to run, which can be installed via `pip`. The `main()` function in `gibbs_sampler.py` has all the parameters of a calculation that can be adjusted at the top.

Additionally, the scripts `gap_vs_u.py` and `slope_vs_nqb.py` each produce plots from the paper with one selected line. The parameters can be changed in the script to produce the other lines of each plot. The runtime for the default values on a regular desktop computer should not exceed a few minutes.

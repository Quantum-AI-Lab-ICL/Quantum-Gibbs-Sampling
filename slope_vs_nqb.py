import numpy as np
import matplotlib.pyplot as plt
from gibbs_sampler import fermi_hubbard_hamiltonian, majorana_coupling_ops, local_pauli_coupling_ops, gaussian_filter, metropolis_filter, GibbsLindbladian, lindblad_gap

# Parameters
t = 1
beta = 1
n_y = 1
spin = False

slopes = []
n_x_list = range(2, 7)
delta_U = 1e-5
for n_x in n_x_list:
    print(f"Working on nx = {n_x}")
    n_qb = n_x * n_y * (2 if spin else 1)
    coupling_ops = majorana_coupling_ops(n_qb)
    filter_func = gaussian_filter(beta, n_diss=coupling_ops.shape[0])
    hamil = fermi_hubbard_hamiltonian(n_x, n_y, t, 0, spin=spin)
    lindbladian = GibbsLindbladian(hamil, beta, coupling_ops, filter_func)
    U_zero_gap = lindblad_gap(lindbladian)
    hamil = fermi_hubbard_hamiltonian(n_x, n_y, t, delta_U, spin=spin)
    lindbladian = GibbsLindbladian(hamil, beta, coupling_ops, filter_func)
    slopes.append((U_zero_gap - lindblad_gap(lindbladian)) / np.abs(delta_U))

plt.plot(n_x_list, slopes)
plt.show()

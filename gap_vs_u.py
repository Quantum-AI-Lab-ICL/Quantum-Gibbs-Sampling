import numpy as np
import matplotlib.pyplot as plt
from gibbs_sampler import fermi_hubbard_hamiltonian, local_pauli_coupling_ops, metropolis_filter, GibbsLindbladian, lindblad_gap

# Parameters
t = 1
beta = 5
n_x = 5
n_y = 1
spin = False
n_qb = n_x * n_y * (2 if spin else 1)
coupling_ops = local_pauli_coupling_ops(n_qb)
filter_func = metropolis_filter(beta, S=10, n_diss=coupling_ops.shape[0])

gaps = []
U_list = np.linspace(-10, 10, 201)
for U in U_list:
    print(f"Working on U = {U}")
    hamil = fermi_hubbard_hamiltonian(n_x, n_y, t, U, spin=spin)
    lindbladian = GibbsLindbladian(hamil, beta, coupling_ops, filter_func)
    gaps.append(lindblad_gap(lindbladian))

plt.plot(U_list, gaps)
plt.show()

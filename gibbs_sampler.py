"""
Code for exact numerical simulations of fermionic quantum Gibbs
samplers for the paper https://arxiv.org/abs/2501.01412.
Author: Richard Meister
Affiliation: Department of Computing, Imperial College London
Created: 2025

Description:
This simple implementation of quantum Gibbs samplers for fermionic
systems was used to generate the numerical results in the publication
arXiv:2501.01412. It takes in parameters of the physical model and
hyperparameters of the sampling algorithm, generates the Lindbladian
of the sampler and calculates its gap, which is directly proportional
to the mixing time.
"""

import numpy as np
import scipy as sp
import warnings
import openfermion as of
from openfermion.hamiltonians import fermi_hubbard
from functools import reduce

def bump(nu):
    nu = np.array(nu)
    f = np.empty(nu.shape)
    nonzero_ind = np.abs(nu) < 1
    f[nonzero_ind] = np.exp(-.2 / (1 - nu[nonzero_ind]**2))
    f[~nonzero_ind] = 0
    return f

def gaussian_filter(beta, S=None, n_diss=1):
    if S is not None:
        return lambda x: np.exp(-(beta * x)**2 / 8) * np.ones((n_diss, ) + x.shape) * bump(x / S)
    return lambda x: np.exp(-(beta * x)**2 / 8) * np.ones((n_diss, ) + x.shape)

def metropolis_filter(beta, S, n_diss=1):
    q = lambda x: np.exp(-np.sqrt(1 + beta**2 * x**2) / 4) * bump(x / S) * np.ones((n_diss, ) + x.shape)
    return q

def lindblad_ops_from_coupling_ops(hamil, beta, coupling_ops, filter_funcs, hamil_eigsys=None, return_steady_state=False):
    if hamil_eigsys is None:
        hamil_eigsys = sp.linalg.eigh(hamil)
    hamil_eigvals, hamil_eigvecs = hamil_eigsys
    nu_mat = hamil_eigvals[:, None] - hamil_eigvals[None, :]
    jump_ops = hamil_eigvecs.T.conj() @ coupling_ops @ hamil_eigvecs
    jump_ops *=  filter_funcs(nu_mat) * np.exp(-beta * nu_mat / 4)
    # Re-writing the equation for G right below Remark 12 in 2404.05998v2 in the Hamiltonian
    # eigenbasis allows this more efficient calculation of the coherent term.
    G = np.swapaxes(jump_ops.conj(), -1, -2) @ jump_ops * (-.5j * np.tanh(-beta * nu_mat / 4))
    jump_ops = hamil_eigvecs @ jump_ops @ hamil_eigvecs.T.conj()
    G = np.sum(hamil_eigvecs @ G @ hamil_eigvecs.T.conj(), axis=0, keepdims=False)
    if return_steady_state:
        steady_state = np.exp(-beta * hamil_eigvals)
        steady_state /= np.sum(steady_state)
        steady_state = np.diag(steady_state)
        steady_state = hamil_eigvecs @ steady_state @ hamil_eigvecs.T.conj()
        return jump_ops, G, steady_state
    return jump_ops, G

def fermi_hubbard_hamiltonian(n_x, n_y, t, U, spin=False):
    hamil = fermi_hubbard(n_x, n_y, t, U, spinless=(not spin), periodic=False)
    qubit_hamil = of.transforms.jordan_wigner(hamil)
    return of.linalg.get_sparse_operator(qubit_hamil).todense()

def local_pauli_coupling_ops(n_qubits):
    hilbert_dim = 2**n_qubits
    coupling_ops = np.empty((3 * n_qubits, hilbert_dim, hilbert_dim), dtype='complex128')
    id = np.array([[1, 0], [0, 1]], dtype='complex128')
    paulis = (np.array([[0, 1], [1, 0]], dtype='complex128'),
              np.array([[0, -1j], [1j, 0]], dtype='complex128'),
              np.array([[1, 0], [0, -1]], dtype='complex128'))
    for k in range(n_qubits):
        for n, pauli in enumerate(paulis):
            local_op_list = [id] * k + [pauli] + [id] * (n_qubits - k - 1)
            coupling_ops[3 * k + n, :, :] = reduce(np.kron, local_op_list)
    return coupling_ops

def majorana_coupling_ops(n_sites):
    majorana_ops = np.empty((2 * n_sites, 2**n_sites, 2**n_sites), dtype='complex128')
    for k in range(n_sites):
        maj_plus = of.ops.FermionOperator(((k, 1),), 1) + of.ops.FermionOperator(((k, 0),), 1)
        maj_minus = of.ops.FermionOperator(((k, 1),), 1j) + of.ops.FermionOperator(((k, 0),), -1j)
        majorana_ops[2 * k, :, :] = of.linalg.get_sparse_operator(of.transforms.jordan_wigner(maj_plus), n_qubits=n_sites).todense()
        majorana_ops[2 * k + 1, :, :] = of.linalg.get_sparse_operator(of.transforms.jordan_wigner(maj_minus), n_qubits=n_sites).todense()
    return majorana_ops


class GibbsLindbladian(sp.sparse.linalg.LinearOperator):

    def __init__(self, hamil, beta, coupling_ops, filter_func, calc_ops=True, global_shift=0, steady_state_shift=0, dtype='complex128'):
        self.hamil = hamil
        self.beta = beta
        self.coupling_ops = coupling_ops
        self.filter_func = filter_func
        self._jump_ops = None
        self._coherent_op = None
        self._steady_state = None
        self._dtype = None
        self._requested_dtype = dtype
        self.global_shift = global_shift
        self.steady_state_shift = steady_state_shift
        self.matvec_calls = 0
        if calc_ops:
            self.calculate_operators()

    @property
    def shape(self):
        return (self.hamil.shape[0]**2,) * 2

    @property
    def dtype(self):
        if self._dtype is None:
            self.calculate_operators()
        return self._dtype

    @property
    def jump_ops(self):
        if self._jump_ops is None:
            self.calculate_operators()
        return self._jump_ops

    @property
    def coherent_op(self):
        if self._coherent_op is None:
            self.calculate_operators()
        return self._coherent_op

    @property
    def steady_state(self):
        if self._steady_state is None:
            self.calculate_operators()
        return self._steady_state

    def as_matrix(self):
        rho = np.empty((self.shape[1],), dtype='complex128')
        lmat = np.empty(self.shape, dtype='complex128')
        for k in range(self.shape[0]):
            rho[:] = 0
            rho[k] = 1
            rho = self @ rho
            lmat[:, k] = rho
        return lmat

    def calculate_operators(self):
        self._jump_ops, coherent_op, self._steady_state = lindblad_ops_from_coupling_ops(self.hamil, self.beta, self.coupling_ops, self.filter_func, return_steady_state=True)
        self._jump_ops = self.jump_ops.astype(self._requested_dtype)
        coherent_op = coherent_op.astype(self._requested_dtype)
        self._steady_state = self._steady_state.reshape((-1)).astype(self._requested_dtype)
        # Re-normalise to vector norm after vectorisation.
        self._steady_state /= np.linalg.norm(self._steady_state)
        # For some reason, this loop is often a lot faster than sum(matmul()).
        Ldag_L = np.zeros(self._jump_ops.shape[1:], dtype=self._jump_ops.dtype)
        for k in range(self._jump_ops.shape[0]):
            Ldag_L += self._jump_ops[k, :, :].T.conj() @ self._jump_ops[k, :, :]
        Ldag_L *= .5
        self._left_fact = -1j * coherent_op - Ldag_L
        self._right_fact = 1j * coherent_op - Ldag_L
        self._left_fact = np.ascontiguousarray(np.real_if_close(self._left_fact, tol=1e3))
        self._right_fact = np.ascontiguousarray(np.real_if_close(self._right_fact, tol=1e3))
        self._steady_state = np.ascontiguousarray(np.real_if_close(self._steady_state, tol=1e3))
        self._jump_ops = np.ascontiguousarray(np.real_if_close(self._jump_ops, tol=1e3))
        self._dtype = np.result_type(self._left_fact, self._right_fact, self._steady_state, self._jump_ops)
        self._steady_state = self._steady_state.reshape((-1))
        # Re-normalise to vector norm after vectorisation.
        self._steady_state /= np.linalg.norm(self._steady_state)
        self._real_jumps = False

    def _matvec(self, rho):
        self.matvec_calls += 1
        dim = int(np.sqrt(rho.size))
        # Contiguous is essential for matrix products to be performant.
        rho_mat = np.ascontiguousarray(rho.reshape((dim, dim)))
        La = self.jump_ops
        Lrho = self._left_fact @ rho_mat
        Lrho += rho_mat @ self._right_fact
        if self._real_jumps:
            for k in range(self._jump_ops_real.shape[0]):
                Lrho += self._jump_ops_real[k, :, :] @ rho_mat @ self._jump_ops_real[k, :, :].T + self._jump_ops_imag[k, :, :] @ rho_mat @ self._jump_ops_imag[k, :, :].T
        else:
            tmparr = np.empty(La.shape[1:])
            # As in calculate_operators(), this loop is faster than sum(matmul()).
            for k in range(La.shape[0]):
                tmparr = La[k, :, :] @ rho_mat
                Lrho += tmparr @ La[k, :, :].T.conj()
                #Lrho += La[k, :, :] @ rho_mat @ La[k, :, :].T.conj()
        Lrho = Lrho.reshape((-1))
        rho_mat = rho_mat.reshape((-1))
        if self.global_shift != 0:
            Lrho += self.global_shift * rho_mat
        if self.steady_state_shift != 0:
            Lrho += self.steady_state_shift * (self.steady_state.T.conj() @ rho_mat) * self.steady_state
        return Lrho


def lindblad_gap(lindbladian):
    max_eigval = sp.sparse.linalg.eigs(lindbladian, k=1, return_eigenvectors=False)
    lindbladian.global_shift = -max_eigval[0]
    lindbladian.steady_state_shift = max_eigval[0]
    shifted_gap = sp.sparse.linalg.eigs(lindbladian, k=1, return_eigenvectors=False)
    lindbladian.global_shift = 0
    lindbladian.steady_state_shift = 0
    gap = -(shifted_gap + max_eigval)
    if np.imag(gap) > 1e-10:
        warnings.warn(f"Found imaginary part of {np.imag(gap)} in gap")
    return gap[0].real

def main():
    # Parameters
    t = 1
    U = 0
    beta = 1
    n_x = 2
    n_y = 2
    spin = False
    hamil = fermi_hubbard_hamiltonian(n_x, n_y, t, U, spin=spin)
    n_qb = n_x * n_y * (2 if spin else 1)
    coupling_ops = majorana_coupling_ops(n_qb)
    filter_func = gaussian_filter(beta, n_diss=coupling_ops.shape[0])

    # Calculation
    lindbladian = GibbsLindbladian(hamil, beta, coupling_ops, filter_func)
    gap = lindblad_gap(lindbladian)
    analytical_gap_str = ""
    if U == 0 and min(n_x, n_y) == 1:
        analytical_gap = 2 * np.exp(-beta**2 * np.cos(np.pi / (n_qb + 1))**2) * np.cosh(beta * np.cos(np.pi / (n_qb + 1)))
        analytical_gap_str = f", analytical gap = {analytical_gap}"
    print(f"gap = {gap}{analytical_gap_str}")

if __name__ == "__main__":
    main()

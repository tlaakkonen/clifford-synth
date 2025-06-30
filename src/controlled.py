from galois import FieldArray, Poly, GF2 as GF, gcd, lcm
import numpy as np

from common import synth_gauss, to_parity_matrix

# This file contains an implementation of controlled CNOT circuit synthesis.
# The main technical component is the calculation of the rational canonical form 
# over GF2, which is done following the method described by M. Geck in:
# > Geck, M., 2020. On Jacob's construction of the rational canonical form of a matrix. 
# > The Electronic Journal of Linear Algebra, 36, pp.177-182. 
# > [DOI: 10.13001/ela.2020.5055](https://doi.org/10.13001/ela.2020.5055)

def krylov_subspace(A: FieldArray, v: FieldArray, minpoly_only: bool = False) -> Poly | tuple[FieldArray, FieldArray, FieldArray, int]:
    """Compute the Krylov subspace K = Span { v, Av, A^2v, ... }
        
    Args:
        A: A square matrix whose image forms the whole vector space
        v: The starting vector for the subspace
        minpoly_only: If True, only return the minimal polynomial of the subspace

    Returns:
        If minpoly_only = False:
            - T: The matrix representing the action of A in the subspace basis
            - bch: A matrix giving the basis of the subspace
            - ibch: A matrix giving the dual basis of the subspace
            - rank: The dimension of the subspace
        If minpoly_only = True:
            - minpoly: The minimal polynomial of the action of A on the subspace
    """

    # Compute the Krylov sequence [v, Av, A^2v, ...]
    krylov = [v]
    for _ in range(A.shape[0]):
        krylov.append(A @ krylov[-1])
    krylov = np.stack(krylov, axis=1)

    # Find the dual basis using Gaussian elimination:
    krylov_augmented = np.concatenate([krylov, GF.Identity(A.shape[0])], axis=1)
    krylov_augmented = krylov_augmented.row_reduce()
    ibch = krylov_augmented[:, A.shape[0] + 1:]

    # Determine the rank of the subspace by finding the first non-pivot:
    for i in range(A.shape[0]):
        if krylov_augmented[i, i] == 0:
            break
    else:
        i = A.shape[0]
    rank = i

    
    if minpoly_only:
        # If minpoly_only, return the minimal polynomial, extracted from
        # the left half of ext, which is its companion matrix.
        minpoly = Poly(np.concatenate([krylov_augmented[:i, i], GF([1])]), order='asc')
        return minpoly

    # Otherwise, find the actual basis by inverting the dual basis and
    # simultaneously transform A by this basis:
    ibch_augmented = np.concatenate([ibch.T, GF.Identity(A.shape[0]), A.T], axis=1)
    ibch_augmented = ibch_augmented.row_reduce()
    # Extract the basis change and the transformed A:
    bch = ibch_augmented[:, A.shape[0]:2*A.shape[0]].T
    Abch = ibch_augmented[:, 2*A.shape[0]:].T
    T = ibch@Abch
    return T, bch, ibch, rank
    
def jacob_complement(T: FieldArray, d: int) -> tuple[FieldArray, FieldArray]:
    """Compute Jacob's block-diagonal basis

    Compute a change of basis for the block upper triangular matrix T so that 
    it is block-diagonal, where the size of the upper left block is dxd. 
    It is required that e_1 is a maximal vector for T.

    Args:
        T: A block upper triangular square matrix, for which e_1 is a maximal vector
        d: The size of the upper left block

    Returns:
        bch: The change of basis matrix
        ibch: The inverse change of basis matrix
    """

    # I don't fully understand why this works, but see Remark 3.3 in Geck for details.

    # Compute the Krylov subspace of T.T starting from e_d:
    v = GF.Zeros(T.shape[0])
    v[d - 1] = 1
    rows = [v]
    for _ in range(1, d):
        rows.append(rows[-1] @ T)
    # Complete the vector space using a basis for its null space:
    extras = np.stack(rows, axis=0).null_space()
    # Find the corresponding basis change:
    bch = np.concatenate([GF.Identity(T.shape[0])[:, :d], extras.T], axis=1)
    ibch = np.linalg.inv(bch)
    return bch, ibch

def eval_matrix_poly(A: FieldArray, v: FieldArray, p: Poly) -> FieldArray:
    """Evaluate p(A) @ v
    
    Args:
        A: A square matrix.
        v: The vector to multiply with the resulting matrix.
        p: The polynomial of the matrix to compute
    
    Returns:
        total: The vector p(A) @ v
    """
    total = GF.Zeros(v.shape)
    z = v.copy()
    for c in p.coefficients(order="asc"):
        total += c * z
        z = A @ z
    return total

def lcm_poly_vector(A: FieldArray, v: FieldArray, mu_v: Poly, w: FieldArray, mu_w: Poly) -> FieldArray:
    """Compute a vector u with minimal polynomial mu_u = lcm(mu_v, mu_w)
    
    Given two vectors v,w and their minimal polynomials mu_v,mu_w with respect to A, compute
    a vector u and its minimal polynomial mu_u such that mu_u = lcm(mu_v, mu_w).

    Args:
        A: The square matrix which the minimal polynomials apply to.
        v: A vector compatible with A
        mu_v: The minimal polynomial of v with respect to A
        w: A vector compatible with A
        mu_w: The minimal polynomial of w with respect to A

    Returns:
        u: A vector such that mu_u = lcm(mu_v, mu_w)
    """
    # If one of mu_w, mu_v is zero, return the non-zero value 
    d = gcd(mu_v, mu_w)
    if d.degree == 0:
        return v + w
    # Otherwise, follow the construction from Lemma 4.4 in Geck: 
    qt = mu_w // d
    h = gcd(d, qt ** d.degree)
    k = d // h
    v2 = eval_matrix_poly(A, v, h)
    w2 = eval_matrix_poly(A, w, k)
    return v2 + w2

def find_maximal_vector(A: FieldArray) -> tuple[FieldArray, Poly]:
    """Compute a maximal vector of A
    
    Given a square matrix A, compute a vector v such that the minimal polynomial
    mu_v of v with respect to A is equal to mu_A, the minimal polynomial of A itself.

    Args:
        A: A square matrix
    
    Returns:
        z: A maximal vector for A
    """

    # This implements Corollary 4.5 in Geck.

    # Start with e = e_1:
    e = GF.Zeros(A.shape[0])
    e[0] = 1
    # Compute the minimal polynomial of e_1
    p = krylov_subspace(A, e, minpoly_only=True)
    # Maintain a list of which standard basis vectors
    # are needed to compute the maximal vector z
    l = [0]
    # We will loop over all standard basis vectors in turn,
    # and compute the lcm of all of their minimal polynomials,
    # which must be equal to the minimal polynomial of A.
    for i in range(1, A.shape[0]):
        # If degree(p) = n, then p is the characteristic polynomial
        # and also the minimal polynomial, so we can exit
        if p.degree == A.shape[0]:
            break
        
        # Compute the next standard basis vector and its minimal polynomial:
        e = GF.Zeros(A.shape[0])
        e[i] = 1
        pe = krylov_subspace(A, e, minpoly_only=True)
        # Compute the LCM of this minimal polynomial with 
        # the cumulative polynomial so far
        pp = lcm(p, pe)
        if pp == pe:
            # If the cumulative polynomial divides this minimal polynomial
            # then we can reset the history trace since we might as
            # as well have started here:
            l = [i]
        elif pp == p:
            # If this minimal polynomial divides the cumulative polynomial
            # then we have learned nothing so continue to the next vector:
            continue
        else:
            # Otherwise they have a non-trivial LCM, so we write down
            # which basis vector was needed to get this LCM:
            l.append((i, p, pe))
        # Update the cumulative polynomial for the next iteration
        p = pp

    # Now we compute the maximal vector using only the LCM computations that were
    # necessary to produce the minimal polynomial, since these are expensive.
    z = GF.Zeros(A.shape[0])
    z[l[0]] = 1
    for i, mu_z, mu_e in l[1:]:
        # For each non-trivial polynomial LCM in the history trace, 
        # compute the corresponding LCM vector
        e = GF.Zeros(A.shape[0])
        e[i] = 1
        z = lcm_poly_vector(A, z, mu_z, e, mu_e)
    assert p == krylov_subspace(A, z, minpoly_only=True)
    return z

def transform_maximal_block(A: FieldArray) -> tuple[FieldArray, FieldArray, int]:
    """Find a change of basis which block diagonalizes A along its maximal vector

    Find a change of basis which transforms A into a block diagonal matrix with two blocks
    where the upper-left block is the companion matrix of the minimal polynomial of A.

    Args:
        A: A square matrix

    Returns:
        bch: The change of basis matrix
        ibch: The inverse change of basis matrix
        d: The size of the upper-left block
    """
    v = find_maximal_vector(A)
    T, bch, ibch, d = krylov_subspace(A, v)
    bch2, ibch2 = jacob_complement(T, d)
    bch_total = bch @ bch2
    bchinv_total = ibch2 @ ibch
    return bch_total, bchinv_total, d

def rational_canonical_form(A: FieldArray) -> tuple[FieldArray, FieldArray, FieldArray, list[int]]:
    """Compute the rational canonical form of a square matrix.

    Args:
        A: A square matrix
    
    Returns:
        T: The rational canonical form of A
        bch: The change of basis matrix that maps A to T
        ibch: The inverse change of basis matrix that maps T to A
        indices: The indices of the blocks along the diagonal of T
    """
    # Implements Remark 3.3 from Geck.

    # Keep track of T, bch, ibch, and the indices. We gradually transform
    # A into T by recursively decomposing the bottom-right block.
    T = A.copy()
    total_k = 0
    indices = [0]
    bch, bchinv = GF.Identity(A.shape[0]), GF.Identity(A.shape[0])
    # While part of A is still untransformed:
    while total_k < A.shape[0]:
        # If the remaining block is the identity matrix, we can stop here.
        if not np.any(T[total_k:, total_k:] + GF.Identity(A.shape[0] - total_k)):
            indices.extend(range(total_k + 1, A.shape[0] + 1))
            break
        
        # Otherwise block diagonalize the remaining bottom right block
        # along its maximal vector:
        bchb, bchbinv, k = transform_maximal_block(T[total_k:, total_k:])
        # Apply the corresponding change of basis to T
        T[total_k:, total_k:] = bchbinv @ T[total_k:, total_k:] @ bchb
        # Update the change of basis matrices by block matrix multiplication:
        bch[total_k:, total_k:] = bch[total_k:, total_k:] @ bchb
        bch[:total_k, total_k:] = bch[:total_k, total_k:] @ bchb
        bchinv[total_k:, total_k:] = bchbinv @ bchinv[total_k:, total_k:]
        bchinv[total_k:, :total_k] = bchbinv @ bchinv[total_k:, :total_k]
        total_k += k
        indices.append(total_k)
    
    # Sanity checks:
    # 1. T is equal to A up to the basis change
    assert not np.any(T + bchinv @ A @ bch)
    mask = np.ones(A.shape, dtype=bool)
    for i, j in zip(indices, indices[1:]):
        mask[i:j, i:j] = False
    # 2. T is block diagonal
    assert not np.any(T[mask])
    # 3. bch and ibch are actually inverses
    assert not np.any(bchinv @ bch + GF.Identity(A.shape[0]))
    
    return T, bch, bchinv, indices

def synthesize_controlled_cnot(igates: list[tuple[int, int]]) -> list[tuple[int, int] | tuple[int, int, int]]:
    """Synthesize a controlled CNOT circuit
    
    Given a list of CNOT gates, output a list of CNOT and Toffoli gates that represent a controlled
    version of the input circuit. The control qubit is given by the index one higher than the maximum
    qubit index in the input circuit.

    Args:
        igates: A list of CNOT gates represented as (control, target)

    Returns:
        gates: A list of CNOT or Toffoli gates represented as (control, target) or (control1, control2, target)
    """

    # Compute the rational canonical form of the parity matrix:
    A = to_parity_matrix(igates)
    qubits = A.shape[0]
    T, bch, _, indices = rational_canonical_form(A)

    # Use Gaussian elimination to synthesize the change of basis:
    gates = synth_gauss(bch)
    # For each block of T:
    for i, j in zip(indices, indices[1:]):
        if j == i + 1:
            continue
        
        # Synthesize the rightmost column of the companion matrix.
        # Find the first non-zero coefficient of the minimal polynomial
        for p in range(i + 1, j):
            if T[p, j - 1] != 0:
                # Collect parities onto the corresponding qubit
                for k in range(p + 1, j):
                    if T[k, j - 1] != 0:
                        gates.append((p, k))
                # Control on that qubit
                gates.append((qubits, i, p))
                for k in reversed(range(p + 1, j)):
                    if T[k, j - 1] != 0:
                        gates.append((p, k))
                break
        
        # Synthesize the controlled cyclic shift
        for k in range(i, j - 1):
            gates += [
                (k + 1, k),
                (qubits, k, k + 1),
                (k + 1, k)
            ]

    gates += synth_gauss(bch)[::-1]

    # Sanity checks:
    B = to_parity_matrix([g[-2:] for g in gates], qubits)
    C = to_parity_matrix([g for g in gates if len(g) == 2], qubits)
    # 1. With the control qubit enabled, the circuit is equal to the input
    assert not np.any(A + B)
    # 2. With the control qubit disabled, the circuit is equal to the identity
    assert not np.any(C + GF.Identity(A.shape[0]))

    return gates


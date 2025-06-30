from galois import FieldArray, GF2 as GF
import numpy as np

from common import synth_gauss, to_parity_matrix

def rank_factorization(A: FieldArray) -> tuple[FieldArray, FieldArray]:
    """Compute the rank factorization of a matrix.
    
    Args:
        A: An nxm matrix
    
    Returns:
        C: An nxr matrix where r is the rank of A such that A = CF
        F: An rxm matrix where r is the rank of A such that A = CF
    """
    # Follows the algorithm on Wikipedia

    # Make sure the matrix is thin:
    if A.shape[0] < A.shape[1]:
        A = A.T
        flipped = True
    else:
        flipped = False

    # Compute the row echelon form:
    f = A.row_reduce()
    # Extract pivot columns:
    pivots = []
    for i in range(min(A.shape[0], A.shape[1])):
        if f[len(pivots), i] == 1:
            pivots.append(i)
    # The rank factorization is given by the pivot columns of A
    c = A[:, pivots]
    # And the corresponding rows of the row echelon form
    f = f[:len(pivots), :]
    assert not np.any(c @ f + A) 
    if flipped:
        return f.T, c.T
    else:
        return c, f

def synth_split_rank(B: FieldArray, n: int, down: bool) -> list[tuple[int, int]]:
    """Synthesize a CNOT circuit implementing the biadjacency matrix B
    
    This method will try to minimize the number of non-local operations by using
    the rank decomposition of B.

    Args:
        B: The biadjacency matrix between two groups of qubits
        n: The start index of the second group of qubits
        down: Whether the biadjacency matrix is targeting the first or second group

    Returns:
        gates: A list of CNOT gates represented as (control, target)
    """
    gates = []
    # Write B as a minimal sum of outer products using rank factorization:
    C, F = rank_factorization(B)
    for i in range(C.shape[1]):
        u = C[:, i]
        v = F[i, :]
        # Compute the fanout circuit that accumulates the control
        # and target parities onto one qubit each:
        ii = np.nonzero(u)[0][0]
        jj = np.nonzero(v)[0][0]
        prep = []
        for k in range(B.shape[0]):
            if u[k] != 1: continue
            if k == ii: continue
            if down:
                prep.append((k, ii))
            else:
                prep.append((ii, k))
        for k in range(B.shape[1]):
            if v[k] != 1: continue
            if k == jj: continue
            if down:
                prep.append((jj + n, k + n))
            else:
                prep.append((k + n, jj + n))
        # Then synthesize the outer product:
        if down:
            gates.extend(prep + [(ii, jj + n)] + prep[::-1])
        else:
            gates.extend(prep + [(jj + n, ii)] + prep[::-1])
    return gates

def block_ldu_fact(A: FieldArray, n: int) -> tuple[FieldArray, FieldArray, FieldArray]:
    """Compute the block LDU factorization of a square matrix

    Computes three matrices L,D,U so that L is block lower triangular, D is block diagonal
    and U is block upper diagonal, and A = LDU. The size of the upper-left block is n.
    It is required that the upper-left block of A is invertible.

    Args:
        A: The input square matrix
        n: The size of the upper-left invertible block

    Returns:
        L: The block lower triangular factor
        D: The block diagonal factor
        U: The block upper triangular factor
    """
    # Following the formula on Wikipedia: en.wikipedia.org/wiki/Block_LU_decomposition
    a, b, c, d = A[:n, :n], A[:n, n:], A[n:, :n], A[n:, n:]
    ainv = np.linalg.inv(a)
    I = GF.Identity(A.shape[0])
    Ia, Zb, Zc, Id = I[:n, :n], I[:n, n:], I[n:, :n], I[n:, n:]
    L = np.concatenate([
        np.concatenate([Ia, Zb], axis=1),
        np.concatenate([c@ainv, Id], axis=1),
    ], axis=0)
    D = np.concatenate([
        np.concatenate([a, Zb], axis=1),
        np.concatenate([Zc, d - c@ainv@b], axis=1),
    ], axis=0)
    U = np.concatenate([
        np.concatenate([Ia, ainv@b], axis=1),
        np.concatenate([Zc, Id], axis=1),
    ], axis=0)
    return L, D, U

def make_ul_inv(A: FieldArray, n: int) -> tuple[FieldArray, FieldArray]:
    """Compute a factorization of A where the upper left block is invertible
    
    Computes a factorization A = UR where U is block upper triangular and 
    the upper left block of R is invertible. The size of the upper left
    block is given by n. A must be invertible.

    Args:
        A: The input invertible square matrix
        n: The size of the upper left block

    Returns:
        U: The block upper triangular factor
        R: The factor with invertible upper left block
    """
    # If the upper left block is already invertible, exit early.
    if np.linalg.matrix_rank(A[:n, :n]) == n:
        return GF.Identity(A.shape[0]), A
    
    # Find the column echelon form of the left half of A:
    c = A[:, :n].T.row_reduce().T
    # Extract the rows of A in the upper left block which
    # are not pivots, and the rows of A in the lower left block
    # which are pivots:
    pivots = []
    npivots = []
    epivots = []
    for i in range(c.shape[0]):
        if len(pivots) >= n:
            break
        if c[i, len(pivots)] == 1:
            pivots.append(i)
            if i >= n:
                epivots.append(i)
        elif i < n:
            npivots.append(i)
    
    # Since A is invertible, the left half of A has full rank
    # and so these two sets must have the same size.
    assert len(npivots) == len(epivots)

    # Perform row operations to increase the rank of the upper
    # left block according to the pivot list. Record the row
    # operations into the block upper triangular factor:
    R = A.copy()
    U = GF.Identity(A.shape[0])
    for (nn, ee) in zip(npivots, epivots):
        U[nn, :] += U[ee, :]
        R[nn, :] += R[ee, :]
    
    return U, R

def block_uldu_fact(A: FieldArray, n: int) -> tuple[FieldArray, FieldArray, FieldArray, FieldArray]:
    """Factorize an invertible square matrix into ULDU form

    Factorizes A = ULDU' where U, U' are block upper triangular, L is 
    block lower triangular, and D is block diagonal. The size of the upper
    left block is n.
    
    Args:
        A: An invertible square matrix
        n: The size of the upper left block

    Returns:
        U: The first upper block triangular factor
        L: The block lower triangular factor
        D: The block diagonal factor
        U': The second upper block triangular factor
    """
    U, R = make_ul_inv(A, n)
    L, D, U2 = block_ldu_fact(R, n)
    return U, L, D, U2

def synth_distributed(A: FieldArray, n: int) -> list[tuple[int, int]]:
    """Synthesize a distributed CNOT circuit corresponding to parity matrix A
    
    Synthesizes a CNOT circuit with parity matrix equal to A such that the 
    number of non-local gates connecting qubits [0, n) with [n, A.shape[0])
    is minimized.

    Args:
        A: An invertible square matrix
        n: The size of the first group of qubits

    Returns:
        gates: A list of CNOT gates represented as (control, target)
    """
    # Factorize the matrix:
    U, L, D, U2 = block_uldu_fact(A, n)
    gates = []
    # Synthesize each block triangular part using rank factorization:
    gates.extend(synth_split_rank(U[:n, n:], n, down=False))
    gates.extend(synth_split_rank(L[n:, :n].T, n, down=True))
    # Synthesize the diagonal blocks using Gaussian elimination:
    gates.extend(synth_gauss(D[:n, :n]))
    gates.extend((i + n, j + n) for i, j in synth_gauss(D[n:, n:]))
    gates.extend(synth_split_rank(U2[:n, n:], n, down=False))
    return gates

def synthesize_distributed_cnot(igates: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Resynthesize a CNOT circuit to minimize the number of nonlocal gates
    
    Synthesizes a CNOT circuit equivalent to the original such that the 
    number of non-local gates connecting qubits [0, n) with [n, q)
    is minimized, where q is the number of qubits.

    Args:
        igates: A list of CNOT gates represented as (control, target)
        n: The size of the first group of qubits

    Returns:
        gates: A list of CNOT gates represented as (control, target)
    """
    A = to_parity_matrix(igates)
    gates = synth_distributed(A, n)
    B = to_parity_matrix(gates, A.shape[0])
    # Sanity check that the new and old circuits are equivalent
    assert not np.any(A + B)
    return gates


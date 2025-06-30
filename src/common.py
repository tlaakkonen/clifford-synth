from galois import FieldArray, GF2 as GF

def synth_gauss(A: FieldArray) -> list[tuple[int, int]]:
    """Synthesize a CNOT circuit from its parity matrix using Gaussian elimination

    Args:
        A: An invertible square matrix

    Returns:
        gates: A list of CNOT gates represented as (control, target)
    """
    gates = []
    A = A.copy()
    for i in range(A.shape[0]):
        if A[i, i] == 0:
            for j in range(i + 1, A.shape[0]):
                if A[j, i] != 0:
                    break
            gates += [(i, j), (j, i), (i, j)]
            A[[i, j], :] = A[[j, i], :]

        for j in range(A.shape[0]):
            if i == j: continue
            if A[j, i] != 0:
                gates.append((i, j))
                A[j, :] += A[i, :]
    return gates

def to_parity_matrix(gates: list[tuple[int, int]], qubits: int = None) -> FieldArray: 
    """Construct a parity matrix from a list of CNOT gates
    
    Args:
        gates: A list of CNOT gates represented as (control, target)
        qubits: The size of the output matrix. If not given the maximum qubit index in `gates` will be used.

    Returns:
        A: An invertible square parity matrix corresponding to `gates`
    """  
    if qubits is None:
        qubits = max((max(g, default=0) for g in gates), default=0) + 1

    A = GF.Identity(qubits)
    for i, j in gates:
        assert i != j
        A[:, i] += A[:, j]
    
    return A
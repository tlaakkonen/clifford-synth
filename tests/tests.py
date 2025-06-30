import sys
sys.path.append("../src")

import galois
import numpy as np
import tqdm

import controlled
import distributed
import common

def random_invertible_matrix(n: int) -> galois.FieldArray:
    # Constant expected time, probability of A being invertible is always more than 0.25 regardless of n
    while True:
        A = galois.GF2.Random((n, n))
        if np.linalg.matrix_rank(A) == n:
            return A
        
for _ in tqdm.trange(1000):
    A = random_invertible_matrix(16)
    gates_gauss = common.synth_gauss(A)
    resynth = distributed.synthesize_distributed_cnot(gates_gauss, 8)
    assert not np.any(A + common.to_parity_matrix(resynth, 16))
print("tested 1000 random distributed circuits")


for _ in tqdm.trange(1000):
    A = random_invertible_matrix(8)
    gates_gauss = common.synth_gauss(A)
    resynth = controlled.synthesize_controlled_cnot(gates_gauss)
    cont_true = [g[-2:] for g in resynth]
    cont_false = [g for g in resynth if len(g) != 3]
    assert not np.any(A + common.to_parity_matrix(cont_true, 8))
    assert not np.any(galois.GF2.Identity(8) + common.to_parity_matrix(cont_false, 8))
print("tested 1000 random controlled circuits")

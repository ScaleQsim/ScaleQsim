import cirq
import qsimcirq
import numpy as np
import time
import sys
def qft_circuit(qubits):
    circuit = cirq.Circuit()
    n = len(qubits)
    for i in range(n):
        circuit.append(cirq.H(qubits[i]))
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.append(cirq.CZ(qubits[j], qubits[i]) ** (angle / np.pi))
    for i in range(n // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[n - i - 1]))
    return circuit

qubits = cirq.LineQubit.range(40)
qft_circuit_16 = qft_circuit(qubits)
print(f'qubits = {qubits}')
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0, max_fused_gate_size=4)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)
#initial_state = np.zeros((2**32,), dtype=np.complex64)


try:
    qsim_simulator.simulate(qft_circuit_16, initial_state=0)
except ValueError as e:
    print(f"Warning: Skipping postprocessing due to numpy reshape error: {e}")
sys.stdout.flush()



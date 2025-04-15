import cirq
import qsimcirq
import numpy as np
import time
import sys

qubits = cirq.LineQubit.range(38)

gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0, max_fused_gate_size=4)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

def run_simulation(circuit, label=""):
    start = time.time()
    result = qsim_simulator.simulate(circuit)
    sys.stdout.flush()
    elapsed = time.time() - start
    print(f'{label} runtime: {elapsed:.6f} seconds.')
    return result

def vqc_circuit(qubits, layers=12):
    circuit = cirq.Circuit()
    n = len(qubits)
    for _ in range(layers):
        for q in qubits:
            circuit.append(cirq.rx(np.random.uniform(0, 2*np.pi))(q))
            circuit.append(cirq.ry(np.random.uniform(0, 2*np.pi))(q))
        for i in range(n - 1):
            circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    return circuit

vqc_circ = vqc_circuit(qubits, layers=12)
run_simulation(vqc_circ, label="VQC")

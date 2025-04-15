
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

def qsvm_circuit(qubits, data_vector):
    circuit = cirq.Circuit()
    # Feature map (angle embedding)
    for i, q in enumerate(qubits):
        if i < len(data_vector):
            circuit.append(cirq.ry(data_vector[i])(q))
    # Simple entanglement
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    return circuit

data = np.random.uniform(0, 2*np.pi, size=len(qubits))
qsvm_circ = qsvm_circuit(qubits, data)
run_simulation(qsvm_circ, label="QSVM")

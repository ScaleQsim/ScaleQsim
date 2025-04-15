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

def qv_circuit(qubits, depth=15):
    circuit = cirq.Circuit()
    n = len(qubits)
    for d in range(depth):
        np.random.shuffle(qubits)
        for i in range(0, n - 1, 2):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            u = cirq.unitary(cirq.rz(lam)) @ cirq.unitary(cirq.ry(theta)) @ cirq.unitary(cirq.rz(phi))
            u_gate = cirq.MatrixGate(u)  # ✅ 이1걸로 되돌리면 OK
            circuit.append(u_gate(qubits[i]))
            circuit.append(cirq.CX(qubits[i], qubits[i+1]))
    return circuit



qv_circ = qv_circuit(qubits, depth=15)
run_simulation(qv_circ, label="QV")

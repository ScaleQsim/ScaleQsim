import cirq
import qsimcirq
import numpy as np
import time
import sys

# 1. 큐비트 설정
qubits = cirq.LineQubit.range(38)

# 2. QSim 시뮬레이터 세팅
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=0, max_fused_gate_size=4)
qsim_simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)

# 3. 시뮬레이션 함수
def run_simulation(circuit, label=""):
    start = time.time()
    result = qsim_simulator.simulate(circuit)
    sys.stdout.flush()
    elapsed = time.time() - start
    print(f'{label} runtime: {elapsed:.6f} seconds.')
    return result

# 4. GHZ 회로 생성 함수
def ghz_circuit(qubits):
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(1, len(qubits)):
        circuit.append(cirq.CX(qubits[0], qubits[i]))
    return circuit

# 5. 회로 생성 및 실행
ghz_circ = ghz_circuit(qubits)
run_simulation(ghz_circ, label="GHZ")

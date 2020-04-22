import qiskit
import numpy
import matplotlib.pyplot as plt
import ray

from entangling_operations import *


def qasm_header(n_qubits):
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)


def qasm_measurements_all_qubits(n_qubits):
    qasm=['']
    for i in range(n_qubits):
        qasm.append('measure q[{0}] -> c[{0}];'.format(i))

    return ''.join(qasm)


def qasm_measuremnts(ancillas):
    qasm = ['']
    for ancilla in ancillas:
        qasm.append('measure q[{0}] -> c[{0}];'.format(ancilla))

    return ''.join(qasm)


def custom_noise_model():
    prob_1 = 0.01  # 1-qubit gate
    prob_2 = 0.03   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = qiskit.providers.aer.noise.errors.depolarizing_error(prob_1, 1)
    error_2 = qiskit.providers.aer.noise.errors.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = qiskit.providers.aer.noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return noise_model


def get_statevector_from_qasm(qasm_circuit):
    n_threads = 2
    backend_options = {"method": "statevector", "zero_threshold": 10e-9, "max_parallel_threads": n_threads,
                       "max_parallel_experiments": n_threads, "max_parallel_shots": n_threads}
    backend = qiskit.Aer.get_backend('statevector_simulator')
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
    result = qiskit.execute(qiskit_circuit, backend, backend_options=backend_options).result()
    statevector = result.get_statevector(qiskit_circuit)
    return statevector


# get the statevector produced by the qasm circuit
def get_measurement_counts(qasm, noise=False, shots=1000):
    backend_options = {"zero_threshold": 10e-9, "max_parallel_threads": 8,
                       "max_parallel_experiments": 8, "max_parallel_shots": 8}
    backend = qiskit.Aer.get_backend('qasm_simulator', )
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
    if noise:
        noise_model = custom_noise_model()
    else:
        noise_model = None
    sim_result = qiskit.execute(qiskit_circuit, backend, backend_options=backend_options, noise_model=noise_model, shots=shots).result()
    state_counts = sim_result.get_counts(qiskit_circuit)

    state_counts_reversed = {}
    for state in state_counts.keys():
        count = state_counts[state]
        state_counts_reversed[state[::-1]] = count

    return state_counts_reversed

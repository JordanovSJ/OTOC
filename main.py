import qiskit
import numpy
import matplotlib.pyplot as plt


# <<<<<<<<<<<< We build our circuit using the QASM language, then compiled it with qiskit>>>>>>>>>>>>>>>

def qasm_header(n_qubits):
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{0}];\ncreg c[{0}];\n'.format(n_qubits)


def qasm_measurements(n_qubits):
    qasm=['']
    for i in range(n_qubits):
        qasm.append('measure q[{0}] -> c[{0}];'.format(i))

    return ''.join(qasm)


# Not used
# this is equivalent to a partial n-qubit controlled NOT gate
def multi_qubit_controlled_rx(angle, control_qubits, target_qubit):
    if not list(control_qubits):
        return ['rx({}) q[{}];\n'.format(angle, target_qubit)]
    else:
        qasm = multi_qubit_controlled_rx(angle / 2, control_qubits[:-1], target_qubit)
        qasm.append('cz q[{}], q[{}];\n'.format(control_qubits[-1], target_qubit))
        qasm += multi_qubit_controlled_rx(-angle / 2, control_qubits[:-1], target_qubit)
        qasm.append('cz q[{}], q[{}];\n'.format(control_qubits[-1], target_qubit))
        return qasm


def max_entangling_gate(angle, control_qubit, target_qubits):
    qasm = []
    for qubit in target_qubits:
        qasm += ['rx({}) q[{}];\n'.format(angle/2, qubit)]
        qasm += ['cz q[{}], q[{}];\n'.format(control_qubit, qubit)]
        qasm += ['rx({}) q[{}];\n'.format(-angle/2, qubit)]
        qasm += ['cz q[{}], q[{}];\n'.format(control_qubit, qubit)]

    return qasm


def entangling_gate(angle, qubits):
    qasm_cnots = []
    qasm_hs = []
    for i in range(1, len(qubits)):
        qasm_cnots.append('cx q[{}], q[{}];\n'.format(qubits[i-1], qubits[i]))
        qasm_hs.append('h q[{}];\n'.format(qubits[i]))

    return qasm_hs + qasm_cnots + ['rz ({}) q[{}];\n'.format(angle, qubits[-1])] + qasm_cnots[::-1] + qasm_hs


def otoc_circuit(angle, A_qubit, B_qubit, ancilla_qubits, max_entangling=True):

    B = ['z q[{}];\n'.format(B_qubit)]
    A = ['x q[{}];\n'.format(A_qubit)]

    if max_entangling:
        # we get the conjugate by changing the sign of the angle parameter and reversing the order of the quantum gates
        U_conj = max_entangling_gate(-angle, A_qubit, ancilla_qubits + [B_qubit])[::-1]
        U = max_entangling_gate(angle, A_qubit, ancilla_qubits + [B_qubit])
    else:
        U_conj = entangling_gate(-angle, [A_qubit] + ancilla_qubits + [B_qubit])[::-1]
        U = entangling_gate(angle, [A_qubit] + ancilla_qubits + [B_qubit])

    return ''.join(U_conj + B + U + A + U_conj + B + U + A)


def custom_noise_model():
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01   # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = qiskit.providers.aer.noise.errors.depolarizing_error(prob_1, 1)
    error_2 = qiskit.providers.aer.noise.errors.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = qiskit.providers.aer.noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return noise_model


# get the statevector produced by the qasm circuit
def get_measurement_probs(qasm, noise=False):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm)
    if noise:
        noise_model = custom_noise_model()
    else:
        noise_model = None
    sim_result = qiskit.execute(qiskit_circuit, backend, noise_model=noise_model, shots=1000).result()
    return sim_result.get_counts(qiskit_circuit)


if __name__ == "__main__":

    max_entangling = True  # this controls if we want to use max. entangling U operation
    noise = True
    A_qubit = 0
    B_qubit = 1
    ancilla_qubits = [2, 3, 4]
    # get initial state (|0>+|1>)|0>
    initial_state_qasm = ('h q[{}];\n'.format(A_qubit))

    qasm_header = qasm_header(2 + len(ancilla_qubits))

    otoc_values = []

    angles = numpy.arange(101)*numpy.pi/100
    for angle in angles:
        otoc_qasm = otoc_circuit(angle, A_qubit, B_qubit, ancilla_qubits, max_entangling=max_entangling)
        qasm = qasm_header + initial_state_qasm + otoc_qasm + initial_state_qasm

        # perform a little state tomography to find the sign of the OTOC
        measurement_probs = get_measurement_probs(qasm + qasm_measurements(2 + len(ancilla_qubits)), noise)
        try:
            p1 = (measurement_probs[''.zfill(len(ancilla_qubits)+2)]/1000)
        except KeyError:
            p1 = 0.000001

        try:
            p2 = (measurement_probs[''.zfill(len(ancilla_qubits) + 1)+'1'] / 1000)
        except KeyError:
            p2 = 0.000001

        qasm += 'h q[0];\n'
        measurement_probs = get_measurement_probs(qasm + qasm_measurements(2 + len(ancilla_qubits)), noise)
        try:
            p3 = (measurement_probs[''.zfill(len(ancilla_qubits) + 2)] / 1000)
        except KeyError:
            p3 = 0.000001

        try:
            p4 = (measurement_probs[''.zfill(len(ancilla_qubits) + 1)+'1'] / 1000)
        except KeyError:
            p4 = 0.000001

        otoc_value = (numpy.arccos((p3 - p1 - p2)/(2*(p1*p2)**0.5)) + numpy.arccos((p4 - p1 - p2)/(2*(p1*p2)**0.5)))/2
        otoc_values.append(otoc_value)

    otoc_values = numpy.array(otoc_values)
    plt.plot(angles, otoc_values)
    plt.ylim(-1.1, 1.1)
    plt.xlabel(r't, [$\pi$]')
    plt.ylabel('OTOC')
    plt.show()
    print('Ciao')

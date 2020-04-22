import qiskit
import numpy
import matplotlib.pyplot as plt
import ray


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


# random lattice, increasing entanglement strength
def random_entangling_1(time, qubits):
    qasm = []
    qasm_conj = []
    n_qubits = len(qubits)
    for i in range(n_qubits*2):
        control = round(numpy.random.rand()*(n_qubits-1))  # get a random qubit
        target = round(numpy.random.rand()*(n_qubits-1))
        # make sure that the target is not the same as the control
        while control == target:
            target = round(numpy.random.rand() * (n_qubits - 1))

        # add random 1 qubit rotations
        rotation_angle = 2 * time * numpy.pi * numpy.random.rand()
        qasm.append('rx ({}) q[{}];\n'.format(rotation_angle, control))
        qasm_conj.append('rx ({}) q[{}];\n'.format(-rotation_angle, control))

        rotation_angle = 2 * time * numpy.pi * numpy.random.rand()
        qasm.append('rx ({}) q[{}];\n'.format(rotation_angle, target))
        qasm_conj.append('rx ({}) q[{}];\n'.format(-rotation_angle, target))

        # random partial entangling gate
        entangling_angle = 2*time*numpy.pi #*numpy.random.rand()  # random angle whose max value scales to 2*Pi with time
        qasm += multi_qubit_controlled_rx(entangling_angle, [control], target)
        qasm_conj += multi_qubit_controlled_rx(-entangling_angle, [control], target)

        # add more random 1 qubit rotations
        rotation_angle = 2 * time * numpy.pi * numpy.random.rand()
        qasm.append('rz ({}) q[{}];\n'.format(rotation_angle, control))
        qasm_conj.append('rz ({}) q[{}];\n'.format(-rotation_angle, control))

        rotation_angle = 2 * time * numpy.pi * numpy.random.rand()
        qasm.append('rz ({}) q[{}];\n'.format(rotation_angle, target))
        qasm_conj.append('rz ({}) q[{}];\n'.format(-rotation_angle, target))

    return ''.join(qasm), ''.join(qasm_conj[::-1])


# random lattice, fixed entanglement strength, increasing number of entangling elements
def random_entangling_2(time, qubits):
    qasm = []
    qasm_conj = []
    n_qubits = len(qubits)
    for i in range(int(time*10*n_qubits)+1):
        control = round(numpy.random.rand()*(n_qubits-1))  # get a random qubit
        target = round(numpy.random.rand()*(n_qubits-1))
        # make sure that the target is not the same as the control
        while control == target:
            target = round(numpy.random.rand() * (n_qubits - 1))

        # add random 1 qubit rotations
        rotation_angle = 2 * numpy.pi * numpy.random.rand()
        qasm.append('rx ({}) q[{}];\n'.format(rotation_angle, control))
        qasm_conj.append('rx ({}) q[{}];\n'.format(-rotation_angle, control))

        rotation_angle = 2 * numpy.pi * numpy.random.rand()
        qasm.append('rx ({}) q[{}];\n'.format(rotation_angle, target))
        qasm_conj.append('rx ({}) q[{}];\n'.format(-rotation_angle, target))

        # random partial entangling gate
        qasm.append('cx q[{}], q[{}];\n'.format(control, target))
        qasm_conj.append('cx q[{}], q[{}];\n'.format(control, target))

        # add more random 1 qubit rotations
        rotation_angle = 2 * numpy.pi * numpy.random.rand()
        qasm.append('rz ({}) q[{}];\n'.format(rotation_angle, control))
        qasm_conj.append('rz ({}) q[{}];\n'.format(-rotation_angle, control))

        rotation_angle = 2 * numpy.pi * numpy.random.rand()
        qasm.append('rz ({}) q[{}];\n'.format(rotation_angle, target))
        qasm_conj.append('rz ({}) q[{}];\n'.format(-rotation_angle, target))

    return ''.join(qasm), ''.join(qasm_conj[::-1])


def max_entangling_gate(angle, qubits):
    qasm = []
    qasm_conj = []
    for i in range(len(qubits[1:])):
        target = qubits[i]
        control = qubits[i-1]

        qasm += ['rx({}) q[{}];\n'.format(angle/2, target)]
        qasm_conj += ['rx({}) q[{}];\n'.format(-angle/2, target)]

        qasm += ['cz q[{}], q[{}];\n'.format(control, target)]
        qasm_conj += ['cz q[{}], q[{}];\n'.format(control, target)]

        qasm += ['rx({}) q[{}];\n'.format(-angle/2, target)]
        qasm_conj += ['rx({}) q[{}];\n'.format(angle/2, target)]

        qasm += ['cz q[{}], q[{}];\n'.format(control, target)]
        qasm_conj += ['cz q[{}], q[{}];\n'.format(control, target)]

    return ''.join(qasm), ''.join(qasm_conj[::-1])


def entangling_operation(angle, qubits):
    qasm_cnots = []
    qasm_hs = []
    for i in range(1, len(qubits)):
        qasm_cnots.append('cx q[{}], q[{}];\n'.format(qubits[i-1], qubits[i]))
        qasm_hs.append('h q[{}];\n'.format(qubits[i]))

    U = qasm_hs + qasm_cnots + ['rz ({}) q[{}];\n'.format(angle, qubits[-1])] + qasm_cnots[::-1] + qasm_hs
    U_conj = qasm_hs + qasm_cnots + ['rz ({}) q[{}];\n'.format(-angle, qubits[-1])] + qasm_cnots[::-1] + qasm_hs

    return ''.join(U), ''.join(U_conj)

import numpy as np
import sys, math, time
import warnings

from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

"""

sources: 
    create_circuit - [github.com/Qiskit/qiskit-iqx-tutorials]


usage:

    backend = BasicAer.get_backend("qasm_simulator")

    q_gen = random_gen(5, backend)

    choice = q_gen.choice(array)

"""

class random_gen():
    def __init__(self, glo_num_qubits, glo_backend):
        self.glo_backend = glo_backend
        self.circuit = self.create_circuit(glo_num_qubits)
        
    def create_circuit(self, num_qubits):
        q = QuantumRegister(num_qubits)
        c = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(q, c)
        circuit.h(q)
        circuit.barrier()
        circuit.measure(q, c)
        return circuit
    
    def rand_int(self, bit_precision=32):
        job = execute(self.circuit, self.glo_backend, shots=bit_precision, memory=True)

        num = int(''.join(job.result().get_memory()), 2)
        
        return num
    
    def choice(self, array):
        mx = len(array)
        selected = array[self.rand_int() % mx]
        return selected

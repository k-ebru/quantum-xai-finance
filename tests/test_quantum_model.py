import pennylane as qml
from pennylane import numpy as np

from src.quantum_model import init_weights, make_circuit


def test_all_encoded_features_affect_readout():
    circuit = make_circuit(4)
    x = np.array([0.5, 1.2, 0.8, 2.0], requires_grad=True)
    weights = init_weights(4)

    gradients = qml.grad(circuit, argnums=0)(x, weights)

    assert np.all(np.abs(gradients) > 1e-6)
    assert -1 <= float(circuit(x, weights)) <= 1

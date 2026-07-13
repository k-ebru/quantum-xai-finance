"""Variational quantum classifier and gradient based explainability for the
stress regime task, built with PennyLane.

The circuit uses angle encoding for the input features and a small hardware
efficient ansatz (BasicEntanglerLayers). The readout averages PauliZ on wires
0 and 1. Their combined causal cones cover all four encoded inputs without
the cost of measuring every wire.
"""

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

RANDOM_STATE = 42
N_LAYERS = 2
TRAIN_STEPS = 60
LEARNING_RATE = 0.1


def build_device(n_qubits: int):
    return qml.device("default.qubit", wires=n_qubits)


def make_circuit(n_qubits: int, n_layers: int = N_LAYERS):
    dev = build_device(n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        observable = qml.Hamiltonian(
            [0.5, 0.5],
            [qml.PauliZ(0), qml.PauliZ(1)],
        )
        return qml.expval(observable)

    return circuit


def init_weights(n_qubits: int, n_layers: int = N_LAYERS):
    rng = np.random.default_rng(RANDOM_STATE)
    values = rng.normal(0, 0.1, size=(n_layers, n_qubits))
    return np.array(values, requires_grad=True)


def train_vqc(circuit, X_train, y_train, n_qubits, n_layers=N_LAYERS,
              steps=TRAIN_STEPS, lr=LEARNING_RATE):
    """Train the VQC with plain gradient descent on a mean squared error loss.

    y_train is expected to be encoded as -1 or 1 to match the PauliZ
    expectation value range, not as 0 or 1.

    Returns (weights, loss_history) instead of just weights, so callers can
    plot the training curve without duplicating the training loop.
    """
    weights = init_weights(n_qubits, n_layers)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    X_fixed = [np.array(x, requires_grad=False) for x in X_train]

    def loss_fn(weights):
        preds = np.array([circuit(x, weights) for x in X_fixed])
        return np.mean((preds - y_train) ** 2)

    loss_history = []
    for step in range(steps):
        weights = opt.step(loss_fn, weights)
        current_loss = float(loss_fn(weights))
        loss_history.append(current_loss)
        if step % 10 == 0:
            print(f"step {step}, loss {current_loss:.4f}")

    return weights, loss_history


def predict_vqc(circuit, X, weights):
    X_fixed = [np.array(x, requires_grad=False) for x in X]
    raw = np.array([circuit(x, weights) for x in X_fixed])
    return onp.asarray(raw > 0, dtype=int), onp.asarray(raw, dtype=float)


def quantum_gradient_attribution(circuit, x, weights):
    """Gradient times input attribution for a single sample.

    Differentiates the state-vector simulation with respect to the encoded
    inputs, then multiplies the gradient elementwise by the input. This is an
    exploratory comparison against SHAP, not a validated quantum XAI method.
    """
    x_diff = np.array(x, requires_grad=True)
    grad_fn = qml.grad(circuit, argnums=0)
    gradient = grad_fn(x_diff, weights)
    return np.array(gradient) * np.array(x)

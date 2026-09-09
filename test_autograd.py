import numpy as np
from Autograd_tensor import tensor
import torch


def test_add_forward():
    a = tensor(np.array([1.0, 2.0]))
    b = tensor(np.array([3.0, 4.0]))

    c = a + b

    assert np.allclose(c.data, np.array([4.0, 6.0]))


def test_mul_forward():
    a = tensor(np.array([2.0, 3.0]))
    b = tensor(np.array([4.0, 5.0]))

    c = a * b

    assert np.allclose(c.data, np.array([8.0, 15.0]))


def test_matmul_forward():
    a = tensor(np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ]))

    b = tensor(np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ]))

    c = a @ b

    expected = np.array([
        [19.0, 22.0],
        [43.0, 50.0]
    ])

    assert np.allclose(c.data, expected)


def test_add_backward():
    a = tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = tensor(np.array([3.0, 4.0]), requires_grad=True)

    c = a + b
    loss = c.sum()

    loss.backward()

    assert np.allclose(a.grad, np.array([1.0, 1.0]))
    assert np.allclose(b.grad, np.array([1.0, 1.0]))


def test_mul_backward():
    a = tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = tensor(np.array([4.0, 5.0]), requires_grad=True)

    c = a * b
    loss = c.sum()

    loss.backward()

    assert np.allclose(a.grad, np.array([4.0, 5.0]))
    assert np.allclose(b.grad, np.array([2.0, 3.0]))


def test_chain_rule():
    a = tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = tensor(np.array([4.0, 5.0]), requires_grad=True)

    c = a * b
    d = c + a
    loss = d.sum()

    loss.backward()

    # d = a*b + a
    # dd/da = b + 1
    # dd/db = a

    assert np.allclose(a.grad, np.array([5.0, 6.0]))
    assert np.allclose(b.grad, np.array([2.0, 3.0]))


def test_broadcast_add_forward():
    a = tensor(np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]))

    b = tensor(np.array([10.0, 20.0, 30.0]))

    c = a + b

    expected = np.array([
        [11.0, 22.0, 33.0],
        [14.0, 25.0, 36.0]
    ])

    assert np.allclose(c.data, expected)


def test_broadcast_add_backward():
    a = tensor(
        np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]),
        requires_grad=True
    )

    b = tensor(
        np.array([10.0, 20.0, 30.0]),
        requires_grad=True
    )

    c = a + b
    loss = c.sum()

    loss.backward()

    assert np.allclose(
        a.grad,
        np.ones((2, 3))
    )

    # Chaque valeur de b est utilisée sur 2 lignes
    assert np.allclose(
        b.grad,
        np.array([2.0, 2.0, 2.0])
    )


def test_broadcast_column_backward():
    a = tensor(
        np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]),
        requires_grad=True
    )

    b = tensor(
        np.array([
            [10.0],
            [20.0]
        ]),
        requires_grad=True
    )

    c = a + b
    loss = c.sum()

    loss.backward()

    # Chaque b[i] est utilisé 3 fois sur sa ligne
    assert np.allclose(
        b.grad,
        np.array([
            [3.0],
            [3.0]
        ])
    )


def test_matmul_backward_against_pytorch():
    a_np = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    b_np = np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    #  Tensor
    a = tensor(a_np, requires_grad=True)
    b = tensor(b_np, requires_grad=True)

    out = a @ b
    loss = out.sum()
    loss.backward()

    # PyTorch
    ta = torch.tensor(
        a_np,
        dtype=torch.float64,
        requires_grad=True
    )

    tb = torch.tensor(
        b_np,
        dtype=torch.float64,
        requires_grad=True
    )

    tout = ta @ tb
    tloss = tout.sum()
    tloss.backward()

    assert np.allclose(out.data, tout.detach().numpy())
    assert np.allclose(a.grad, ta.grad.numpy())
    assert np.allclose(b.grad, tb.grad.numpy())


def test_complex_expression_against_pytorch():
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([4.0, 5.0, 6.0])

    # Ton Tensor
    a = tensor(a_np, requires_grad=True)
    b = tensor(b_np, requires_grad=True)

    out = (a * b + a).sum()
    out.backward()

    # PyTorch
    ta = torch.tensor(
        a_np,
        dtype=torch.float64,
        requires_grad=True
    )

    tb = torch.tensor(
        b_np,
        dtype=torch.float64,
        requires_grad=True
    )

    tout = (ta * tb + ta).sum()
    tout.backward()

    assert np.allclose(a.grad, ta.grad.numpy())
    assert np.allclose(b.grad, tb.grad.numpy())



   

import numpy as np

from nanograd import Tensor
from model import GPT, cross_entropy


# Verify the autograd engine produces finite gradients on a simple matmul example.
def test_gradients():
    a = Tensor([[1., 2.], [3., 4.]], True)
    b = Tensor([[2.], [1.]], True)

    loss = ((a @ b) ** 2).mean()
    loss.backward()

    assert np.all(np.isfinite(a.grad))
    assert np.all(np.isfinite(b.grad))
    assert np.any(a.grad != 0)

    print("PASS: gradient calculation")


# Check that a tiny GPT model can run a forward pass, backward pass, and generation.
def test_gpt():
    np.random.seed(7)

    model = GPT(
        vocab_size=12,
        context=8,
        dim=8,
        heads=2,
        layers=1,
    )

    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, 3, 4, 5]])

    logits = model(x)
    assert logits.shape == (1, 4, 12)

    loss = cross_entropy(logits, y)
    loss.backward()

    assert np.isfinite(loss.item())
    assert any(
        p.grad is not None and np.any(p.grad != 0)
        for p in model.parameters()
    )

    generated = model.generate([1, 2], tokens=5)
    assert len(generated) == 7

    print("PASS: GPT forward, backward, loss, generation")


# A mismatched parameter list should fail fast instead of silently corrupting weights.
def test_state_dict_mismatch():
    model = GPT(vocab_size=10, context=4, dim=8, heads=2, layers=1)

    bad_state = [np.zeros_like(p.data) for p in model.parameters()[:-1]]
    try:
        model.load_state_dict(bad_state)
        raise AssertionError("Expected ValueError for mismatched parameter count")
    except ValueError:
        pass

    print("PASS: state dict validation")


if __name__ == "__main__":
    test_gradients()
    test_gpt()
    test_state_dict_mismatch()
    print("All basic tests passed.")

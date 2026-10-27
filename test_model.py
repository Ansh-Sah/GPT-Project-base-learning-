
import numpy as np

from nanograd import Tensor
from model import GPT, cross_entropy


def test_gradients():
    a = Tensor([[1., 2.], [3., 4.]], True)
    b = Tensor([[2.], [1.]], True)

    loss = ((a @ b) ** 2).mean()
    loss.backward()

    assert np.all(np.isfinite(a.grad))
    assert np.all(np.isfinite(b.grad))
    assert np.any(a.grad != 0)

    print("PASS: gradient calculation")


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


if __name__ == "__main__":
    test_gradients()
    test_gpt()
    print("All basic tests passed.")

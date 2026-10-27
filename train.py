
import os
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from model import GPT, cross_entropy


# Training configuration for the small character-level language model.
@dataclass
class TrainConfig:
    context: int = 24
    dim: int = 16
    heads: int = 2
    layers: int = 1
    batch_size: int = 4
    steps: int = 300
    learning_rate: float = 0.001
    eval_every: int = 50
    seed: int = 42
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.context <= 0:
            raise ValueError("context must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


CONFIG = TrainConfig()


# Adam optimizer updates parameters using first and second gradient moments.
class Adam:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.step_count = 0

    # Clear old gradients before computing the next backward pass.
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)

    # Update each parameter using Adam's adaptive learning-rate rule.
    def step(self):
        self.step_count += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = np.clip(p.grad, -1.0, 1.0)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.step_count)
            v_hat = self.v[i] / (1 - self.beta2**self.step_count)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Save the current training state so a run can resume later if needed.
def save_checkpoint(path, model, chars, config, best_val, step):
    np.savez_compressed(
        path,
        state=np.array(model.state_dict(), dtype=object),
        chars=np.array(chars),
        config_json=json.dumps(asdict(config)),
        best_val=np.array(best_val, dtype=np.float64),
        step=np.array(step, dtype=np.int64),
    )


# Create training pairs: each input sequence predicts the next character.
def get_batch(data, batch_size, context):
    starts = np.random.randint(
        0, len(data) - context, size=batch_size
    )
    x = np.stack([data[s:s + context] for s in starts])
    y = np.stack([data[s + 1:s + context + 1] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)


# Train the model on a text corpus and save the best checkpoint.
def main():
    np.random.seed(CONFIG.seed)
    os.makedirs("checkpoints", exist_ok=True)

    with open("data/input.txt", encoding="utf-8") as f:
        text = f.read()

    if len(text) < CONFIG.context + 20:
        raise ValueError("Add more training text to data/input.txt")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)

    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    if len(val_data) <= CONFIG.context + 1:
        train_data, val_data = data, data

    model = GPT(
        len(chars),
        CONFIG.context,
        CONFIG.dim,
        CONFIG.heads,
        CONFIG.layers,
    )

    params = model.parameters()
    optimizer = Adam(params, CONFIG.learning_rate)
    best_val = float("inf")

    print("Training NumPy GPT")
    print("Vocabulary:", len(chars))
    print("Parameters:", sum(p.data.size for p in params))
    print("Training characters:", len(train_data))
    print()

    start = time.time()

    for step in range(1, CONFIG.steps + 1):
        x, y = get_batch(train_data, CONFIG.batch_size, CONFIG.context)

        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()

        for p in params:
            if p.grad is not None:
                p.grad = np.clip(p.grad, -CONFIG.grad_clip, CONFIG.grad_clip)

        optimizer.step()

        if step % CONFIG.eval_every == 0 or step == 1:
            vx, vy = get_batch(val_data, CONFIG.batch_size, CONFIG.context)
            val_loss = cross_entropy(model(vx), vy).item()

            print(
                f"Step {step:4d}/{CONFIG.steps} | "
                f"train loss: {loss.item():.4f} | "
                f"validation loss: {val_loss:.4f} | "
                f"time: {time.time() - start:.1f}s"
            )

            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    "checkpoints/model.npz",
                    model,
                    chars,
                    CONFIG,
                    best_val,
                    step,
                )

    print("\nTraining complete.")
    print("Best validation loss:", best_val)
    print("Checkpoint: checkpoints/model.npz")


if __name__ == "__main__":
    main()

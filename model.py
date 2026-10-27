
import numpy as np
from nanograd import Tensor, softmax, log_softmax


# Create a trainable parameter tensor with a small random initialization.
def parameter(shape, scale=0.02):
    return Tensor(np.random.randn(*shape) * scale, True)


# Dense layer: transforms each token vector with a learned weight matrix.
class Linear:
    def __init__(self, input_dim, output_dim):
        self.w = parameter(
            (input_dim, output_dim),
            (2 / (input_dim + output_dim)) ** 0.5
        )
        self.b = Tensor(np.zeros(output_dim), True)

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


# Lookup table that turns integer token ids into learned vectors.
class Embedding:
    def __init__(self, count, dim):
        self.weight = parameter((count, dim))

    def __call__(self, ids):
        return self.weight[ids]

    def parameters(self):
        return [self.weight]


# Normalize each token embedding across its feature dimension.
class LayerNorm:
    def __init__(self, dim):
        self.gamma = Tensor(np.ones(dim), True)
        self.beta = Tensor(np.zeros(dim), True)

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) * (var + 1e-5) ** -0.5 * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


# Self-attention block: each token can look at earlier tokens in the sequence.
class Attention:
    def __init__(self, dim, heads):
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = Linear(dim, dim * 3)
        self.proj = Linear(dim, dim)

    def __call__(self, x):
        batch, time, dim = x.shape
        qkv = self.qkv(x)

        q = qkv[:, :, :dim]
        k = qkv[:, :, dim:2 * dim]
        v = qkv[:, :, 2 * dim:]

        def split(t):
            return t.reshape(
                batch, time, self.heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (
            self.head_dim ** -0.5
        )

        mask = np.triu(
            np.ones((time, time), dtype=bool), k=1
        )
        scores = scores + Tensor(
            np.where(mask, -1e9, 0.0)
        )

        weights = softmax(scores, axis=-1)
        output = weights @ v

        output = output.transpose(
            0, 2, 1, 3
        ).reshape(batch, time, dim)

        return self.proj(output)

    def parameters(self):
        return self.qkv.parameters() + self.proj.parameters()


# Feed-forward network used after each attention layer.
class MLP:
    def __init__(self, dim):
        self.fc1 = Linear(dim, dim * 4)
        self.fc2 = Linear(dim * 4, dim)

    def __call__(self, x):
        return self.fc2(self.fc1(x).relu())

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


# One transformer block: attention + feed-forward with residual connections.
class Block:
    def __init__(self, dim, heads):
        self.ln1 = LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.mlp.parameters()
        )


# Minimal GPT-style language model that predicts the next token from context.
class GPT:
    def __init__(
        self, vocab_size, context=24,
        dim=16, heads=2, layers=1
    ):
        self.vocab_size = vocab_size
        self.context = context
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = Embedding(context, dim)
        self.blocks = [
            Block(dim, heads) for _ in range(layers)
        ]
        self.final_norm = LayerNorm(dim)
        self.head = Linear(dim, vocab_size)

    def __call__(self, ids):
        batch, time = ids.shape

        x = self.token_emb(ids) + self.pos_emb(
            np.arange(time)
        )

        for block in self.blocks:
            x = block(x)

        return self.head(self.final_norm(x))

    def parameters(self):
        result = (
            self.token_emb.parameters()
            + self.pos_emb.parameters()
        )

        for block in self.blocks:
            result += block.parameters()

        return (
            result
            + self.final_norm.parameters()
            + self.head.parameters()
        )

    def state_dict(self):
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, state):
        params = self.parameters()

        if len(params) != len(state):
            raise ValueError("Parameter count mismatch")

        for p, value in zip(params, state):
            if p.shape != value.shape:
                raise ValueError("Parameter shape mismatch")
            p.data[...] = value

    # Autoregressively sample new ids from the model, one token at a time.
    def generate(
        self, ids, tokens=100,
        temperature=0.8, top_k=10
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        ids = list(ids)

        for _ in range(tokens):
            context_ids = ids[-self.context:]
            x = np.array(
                context_ids, dtype=np.int64
            )[None, :]

            logits = self(x)[0, -1].data
            scores = logits / temperature

            if 0 < top_k < self.vocab_size:
                threshold = np.partition(
                    scores, -top_k
                )[-top_k]
                scores = np.where(
                    scores < threshold, -1e9, scores
                )

            scores = scores - scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()

            next_id = np.random.choice(
                self.vocab_size, p=probabilities
            )
            ids.append(int(next_id))

        return ids


# Compute the average negative log-likelihood of the target tokens.
def cross_entropy(logits, targets):
    batch, time, vocab = logits.shape

    flat = logits.reshape(batch * time, vocab)
    log_probs = log_softmax(flat, axis=-1)

    selected = log_probs[
        np.arange(batch * time),
        targets.reshape(-1)
    ]

    return -selected.mean()

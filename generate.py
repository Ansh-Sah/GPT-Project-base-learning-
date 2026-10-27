
import argparse
import json
import numpy as np

from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.tokens < 0:
        raise ValueError("Token count cannot be negative")

    checkpoint = np.load(
        "checkpoints/model.npz", allow_pickle=True
    )

    chars = checkpoint["chars"].tolist()
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    config = json.loads(str(checkpoint["config_json"]))

    unknown = [ch for ch in args.prompt if ch not in stoi]
    if unknown:
        raise ValueError(
            "Prompt contains characters absent from training data"
        )

    model = GPT(
        len(chars),
        config["context"],
        config["dim"],
        config["heads"],
        config["layers"],
    )

    model.load_state_dict(list(checkpoint["state"]))

    ids = [stoi[ch] for ch in args.prompt]

    result = model.generate(
        ids,
        tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("".join(itos[i] for i in result))


if __name__ == "__main__":
    main()


import argparse
import json
import os

import numpy as np

from model import GPT


# Load a saved GPT checkpoint and sample text from a prompt.
def main():
    # Parse generation settings from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.tokens < 0:
        raise ValueError("Token count cannot be negative")
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top_k cannot be negative")

    checkpoint_path = "checkpoints/model.npz"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    if "chars" not in checkpoint or "config_json" not in checkpoint or "state" not in checkpoint:
        raise ValueError("Checkpoint file is missing required data")

    chars = checkpoint["chars"].tolist()
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    config = json.loads(str(checkpoint["config_json"]))

    required_keys = ["context", "dim", "heads", "layers"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Checkpoint config missing keys: {missing}")

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

"""Download the model weights into a directory (seed a node's model volume).

Ships in the node image so the deployer can populate a named model volume on any device
with a one-shot container (``mixup-node-fetch``), without needing the model copied over the
network from another host.
"""

import argparse

from huggingface_hub import snapshot_download

from mixup_core.constants import HF_MODEL_ID, MODEL_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the model into a directory.")
    parser.add_argument("--model", default=HF_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--dir", default=str(MODEL_DIR), help="destination directory")
    args = parser.parse_args(argv)

    path = snapshot_download(repo_id=args.model, local_dir=args.dir)
    print(f"Downloaded {args.model} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

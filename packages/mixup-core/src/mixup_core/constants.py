"""Constants shared by the host and the nodes: network, ports, and model paths."""

from pathlib import Path

# Docker network and image
NETWORK_NAME = "cluster"
IMAGE_NAME = "mixup"
IMAGE_TAG = "latest"

# Ports
INTERNAL_PORT = 8000        # every node listens here inside the cluster (HTTP control plane)
EXPERT_PORT = 8001          # raw-TCP expert data plane (internal-only; no host mapping)
EXTERNAL_PORT_BASE = 9000   # central -> 9000 on the host, expert containers -> 9001, ...

# Model location inside the container (the host's models/ dir is mounted read-only here)
MODEL_ROOT = Path("models")
MODEL_NAME = "granite-3.1-1b-a400m-instruct"
HF_MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
MODEL_DIR = MODEL_ROOT / MODEL_NAME
MODEL_FILE = MODEL_DIR / "model.safetensors"

# Shared read-write ledger volume (the host's var/ledger dir is mounted read-write here)
LEDGER_DIR = Path("ledger")
CHECKPOINT_FILE = LEDGER_DIR / "checkpoint.json"


def address(container: str) -> str:
    """HTTP URL of a node on the Docker network (container names are hostnames there)."""
    return f"http://{container}:{INTERNAL_PORT}"

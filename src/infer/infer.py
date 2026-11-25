"""Runs inside every container: loads this container's part of the model.

The central container holds everything except the experts (embeddings, attention,
routers, final norm). Each expert container holds a few experts for every layer.
Which container is which comes from the CONTAINER_NAME and CONFIG environment
variables that distribute.py sets.
"""

import copy
import json
import os
import pathlib
import time

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

import requests
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import load, save
from transformers import AutoModelForCausalLM, AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeDecoderLayer, GraniteMoeExperts

NAME: str | None = os.getenv("CONTAINER_NAME")
CONFIG: dict[str, Any] = json.loads(os.environ["CONFIG"])  # {"central": name, "experts": {name: [expert numbers]}}
MODEL_DIR = pathlib.Path("models/granite-3.1-1b-a400m-instruct")
MODEL_FILE = MODEL_DIR / "model.safetensors"

config = GraniteMoeConfig.from_pretrained(MODEL_DIR)
NUM_LAYERS: int = config.num_hidden_layers  # 24
NUM_EXPERTS: int = config.num_local_experts or 0  # 32 per layer (typed Optional in transformers)

SKIP = 999
"""Slot number for experts held by another container. transformers skips any slot number
past the end of a container's experts, so these picks do nothing here."""

Dispatch = Callable[[int, Tensor, Tensor, Tensor], Tensor]
"""(layer, hidden_states, top_k_index, top_k_weights) -> expert output. See ExpertSlot."""

def empty_model() -> GraniteMoeForCausalLM:
    """The whole transformers model, with no weights yet.

    Built on the meta device, so every tensor has a shape but no data and it takes no memory.
    """
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    return cast(GraniteMoeForCausalLM, model)  # from_config picks the class from config.json


class ExpertSlot(torch.nn.Module):
    """Sits where a layer's experts were, on the central container.

    transformers calls it after the router has picked experts. It hands the work to
    `dispatch`, which decides where the experts actually run.
    """

    def __init__(self, layer: int, dispatch: Dispatch) -> None:  # IF CENTRAL, SEND OVER TO SOMEONE, IF SELF, RUN FORWARD PASS
        super().__init__()
        self.layer = layer
        self.dispatch = dispatch

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        """Run this layer's experts through `dispatch`.

        hidden_states:  [tokens, 1024] bfloat16
        top_k_index:    [tokens, 8]    int64, the experts the router picked per token
        top_k_weights:  [tokens, 8]    bfloat16, how much each picked expert counts
        returns:        [tokens, 1024] bfloat16, weighted sum of each token's experts
        """
        return self.dispatch(self.layer, hidden_states, top_k_index, top_k_weights)


def load_central(dispatch: Dispatch) -> GraniteMoeForCausalLM:
    """The central container's model: everything except the experts, ready to generate.

    Every layer's experts are swapped for an ExpertSlot that calls `dispatch`.
    """
    model = empty_model()
    layers = cast(list[GraniteMoeDecoderLayer], model.model.layers)
    for i, layer in enumerate(layers):
        setattr(layer.block_sparse_moe, "experts", ExpertSlot(i, dispatch))

    with safe_open(MODEL_FILE, framework="pt") as f:
        state: dict[str, Tensor] = {
            k.replace("router.layer.weight", "router.weight"): f.get_tensor(k)  # file uses the old name
            for k in f.keys()
            if "input_linear" not in k and "output_linear" not in k  # experts aren't read
        }
    model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()  # the output layer reuses the embeddings
    model.model.rotary_emb = type(model.model.rotary_emb)(config)  # computed, not stored in the file
    return model.eval()


def empty_experts(num_experts: int) -> GraniteMoeExperts:
    """transformers' own experts module, sized for `num_experts` instead of all 32. No weights yet."""
    small_config = copy.deepcopy(config)
    small_config.num_local_experts = num_experts
    small_config._experts_implementation = "grouped_mm"  # transformers' fast path (a full model picks this itself)
    with torch.device("meta"):
        experts = GraniteMoeExperts(small_config)
    # Expert numbers >= num_experts now mean "on another container": skip them instead of crashing.
    setattr(experts, "_is_expert_parallel", True)
    return experts


EXPERT_CONTAINERS: list[str] = list(CONFIG["experts"])


def build_lut(containers: list[str]) -> dict[int, tuple[str, int]]:
    """Expert number -> (container holding it, its slot there), built with // and %.

    e.g. 4 containers of 8: LUT[9] = ("container_3", 1). Change layout with LUT.update(build_lut(new_list)).
    """
    per = NUM_EXPERTS // len(containers)
    return {e: (containers[e // per], e % per) for e in range(NUM_EXPERTS)}


LUT = build_lut(EXPERT_CONTAINERS)


class ExpertGroup:
    """One expert container: holds only its own experts, for every layer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.experts = [e for e, (container, _) in LUT.items() if container == name]  # e.g. [8, ..., 15]

        with safe_open(MODEL_FILE, framework="pt") as file:
            self.layers: list[GraniteMoeExperts] = [
                self.load_layer(file, layer) for layer in range(NUM_LAYERS)
            ]

    def load_layer(self, file: safe_open, layer: int) -> GraniteMoeExperts:
        """Read this container's experts for one layer out of the big file."""
        prefix = f"model.layers.{layer}.block_sparse_moe"
        all_gate_up = file.get_slice(f"{prefix}.input_linear.weight")  # [32, 1024, 1024], nothing read yet
        all_down = file.get_slice(f"{prefix}.output_linear.weight")  # [32, 1024, 512], nothing read yet

        mine = slice(self.experts[0], self.experts[-1] + 1)  # one consecutive block, e.g. 8:16
        experts = empty_experts(len(self.experts))
        experts.load_state_dict({
            "gate_up_proj": all_gate_up[mine],  # reads only ours: [8, 1024, 1024]
            "down_proj": all_down[mine],  # [8, 1024, 512]
        }, assign=True)
        return experts.eval()

    @torch.no_grad()
    def forward(self, layer: int, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        """Run this container's share of one layer's experts.

        Takes the same tensors as ExpertSlot.forward. Tokens that picked none of this
        container's experts get zeros, so adding up every container's result gives the full answer.
        """
        mine = torch.tensor([LUT[e][0] == self.name for e in range(NUM_EXPERTS)])[top_k_index]
        slot = torch.tensor([LUT[e][1] for e in range(NUM_EXPERTS)])[top_k_index]
        slots = torch.where(mine, slot, SKIP)  # real numbers -> our slots
        weights = top_k_weights.masked_fill(~mine, 0)  # other containers' experts count for nothing here
        return self.layers[layer](hidden_states, slots, weights)


PORT = 8000
"""Every expert container listens here (INTERNAL_PORT in distribute.py)."""


def address(container: str) -> str:
    """URL of an expert container on the Docker network (container names are hostnames there)."""
    return f"http://{container}:{PORT}"


# --- central -------------------------------------------------------------------

POOL = ThreadPoolExecutor(max_workers=len(EXPERT_CONTAINERS))
"""Lets dispatch wait on every expert container at once instead of one after another."""


def send(container: str, layer: int, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
    """Ask `container` to run its experts for one layer. Returns its [tokens, 1024] share."""
    body = save({
        "hidden_states": hidden_states.contiguous(),
        "top_k_index": top_k_index.contiguous(),
        "top_k_weights": top_k_weights.contiguous(),
    })
    response = requests.post(f"{address(container)}/{layer}", data=body)
    response.raise_for_status()
    return load(response.content)["out"]


def dispatch(layer: int, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
    """Send one layer's expert work to the containers that hold the picked experts, and add up their replies."""
    jobs = [
        POOL.submit(send, container, layer, hidden_states, top_k_index, top_k_weights)
        for container in {LUT[e][0] for e in top_k_index.flatten().tolist()}
    ]
    total = torch.zeros_like(hidden_states)
    for job in jobs:
        total += job.result()
    return total


def wait_for_experts() -> None:
    """Block until every expert container answers (they take a few seconds to load their weights)."""
    for container in EXPERT_CONTAINERS:
        while True:
            try:
                requests.get(address(container), timeout=2)  # any reply, even a 404, means it's up
                break
            except requests.ConnectionError:
                time.sleep(1)
        print(f"{container} is up", flush=True)


def server_central() -> None:
    """Start the central container: load the model, wait for the experts, generate."""
    model = load_central(dispatch)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    wait_for_experts()

    prompt = os.getenv("PROMPT", "What is mixture of experts?")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(out[0][ids.shape[1]:]), flush=True)


# --- expert containers -----------------------------------------------------------

def serve_experts(group: ExpertGroup, port: int) -> None:
    """Answer expert requests: POST /<layer> with the safetensors bytes that send() makes."""
    app = FastAPI()

    @app.post("/{layer}")
    async def run(layer: int, request: Request) -> Response:
        x = load(await request.body())
        out = group.forward(layer, x["hidden_states"], x["top_k_index"], x["top_k_weights"])
        return Response(save({"out": out}))

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def server_expert() -> None:
    """Start an expert container: load its experts and answer requests."""
    assert NAME is not None
    group = ExpertGroup(NAME)
    print(f"{NAME}: experts {group.experts[0]}-{group.experts[-1]} ready on port {PORT}", flush=True)
    serve_experts(group, PORT)


def start() -> None:
    """Start as the central or an expert container, depending on CONTAINER_NAME."""
    if CONFIG["central"] == NAME:
        server_central()
    else:
        server_expert()


if __name__ == "__main__":
    start()

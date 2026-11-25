# Math

Everything here is the Granite MoE forward pass, written out and then mapped onto the
distributed execution. Shapes are for `granite-3.1-1b-a400m-instruct`.

## Model constants

| Symbol | Value | Meaning |
|---|---|---|
| `L` | 24 | number of layers (`num_hidden_layers`) |
| `E` | 32 | experts per layer (`num_local_experts`) |
| `k` | 8 | experts chosen per token (top-k routing) |
| `d` | 1024 | hidden size (`hidden_size`) |
| `m` | 512 | expert intermediate size (`intermediate_size`) |
| `N` | 4 | number of expert containers |
| `g` | 8 | experts per container (`E / N`) |

## Dense transformer part (runs on the central)

For a hidden state `x ∈ R^{t×d}` (t tokens) at layer `l`:

1. Layer norm and self-attention produce `x' ∈ R^{t×d}` (Granite uses RMSNorm / rotary
   embeddings, but these are opaque to us — transformers computes them).
2. The MoE block replaces the FFN (see below) and yields `x'' ∈ R^{t×d}`.
3. Residual add + final norm + LM head produce logits for the next token.

## MoE block (per layer, per token)

### 1. Router (gate)

The router is a single linear map with weights `W_r ∈ R^{E×d}` (and optional bias):

```
z = W_r · x        # z ∈ R^E, logits over experts
```

Select the top-k experts and softmax their logits:

```
idx     = argsort(z, descending=True)[:k]        # chosen expert indices, idx ∈ R^k
w       = softmax(z[idx])                         # routing weights, w ∈ R^k
```

In the code, `idx` is `top_k_index` and `w` is `top_k_weights`, both produced by
transformers' router before it calls the experts module.

### 2. Expert FFN

Each expert `E_i` is a small MLP (gate/up with SiLU gating, then down-projection):

```
g, u  = split(W_gateup[i] · x)        # W_gateup[i] ∈ R^{2m×d}, g,u ∈ R^m
a     = silu(g) ⊙ u                   # activation, a ∈ R^m
E_i(x) = W_down[i] · a                # W_down[i] ∈ R^{d×m}
```

### 3. Weighted sum

The MoE block output is the routing-weighted sum of the top-k expert outputs:

```
y = Σ_{i ∈ idx} w_i · E_i(x)          # y ∈ R^d
```

## Distributed factorization

The sum above is what gets split across containers. Let the LUT map each global expert
`e` to a container `c(e)` and a local slot `s(e)` (see [routing.md](routing.md)). Then:

```
y = Σ_{j ∈ containers} y_j

y_j = Σ_{i ∈ idx : c(i)=j} w_i · E_i(x)
```

i.e. each container `j` computes the partial sum over the experts *it* holds and
returns `y_j`; the central adds the partials. Because the `c(i)=j` predicate partitions
the chosen experts, the sum is exact.

## How a container computes its partial (the masking)

Container `j` owns `g` experts and instantiates a `GraniteMoeExperts` sized for `g`
local slots (indices `0..g-1`). It must map a *global* chosen index `i` to a *local*
slot, and ignore experts it doesn't own.

Define, for every global expert `e ∈ 0..E-1`:

```
mine[e] = (c(e) == j)                    # bool: do I own expert e?
slot[e] = s(e)                            # local slot if I own it, else undefined
```

For a concrete routing decision `idx, w`, the container builds local tensors:

```
slots   = where(mine[idx], slot[idx], SKIP)     # SKIP = 999, out-of-range sentinel
weights = w ⊙ mine[idx]                         # zero out experts I don't own
y_j     = Experts_l(hidden_states, slots, weights)
```

Two independent mechanisms make non-owned experts contribute zero:

1. **`weights` is masked to 0** for non-owned experts, so their `w_i E_i(x)` term is 0.
2. **`slots` is set to `SKIP = 999`**, which is `≥ g`, so transformers' expert-parallel
   fast path skips that slot entirely (no compute). The flag that enables this is
   `_is_expert_parallel = True` on the experts module.

`SKIP` has to be `≥ g`; `999` is just an arbitrarily large safe value.

## Why the experts module is built "small"

`empty_experts` builds a `GraniteMoeExperts` with `num_local_experts = g` (not `E`),
and sets `_experts_implementation = "grouped_mm"` (transformers' batched-matmul fast
path). This is what lets a container run only 8 experts' matmuls per layer instead of
32, and it's why `slot` indices `≥ g` are "skipped" — they're out of range of a
`g`-sized expert table.

## Complexity (informal)

Per token per layer, with `d=1024, m=512, k=8, g=8`:

- **Dense** (central): attention ≈ `O(t·d²)`, router ≈ `O(E·d)`, norms ≈ `O(d)`.
- **Experts** (distributed): each contacted container does `O(k·d·m)` matmul work, but
  only on its `g` experts. Total expert FLOPs are the same as a monolithic forward; the
  distribution parallelizes them across containers at the cost of one HTTP round-trip
  per layer per generation step.
- **Communication** (central ↔ experts): per layer, `t·k` routing indices (int64) plus
  `t·d` hidden states (bf16), plus the `t·d` partial back — serialized as safetensors.
  For `t=1` this is a few KB per request; the round-trip latency dominates.

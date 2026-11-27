# Implementation Heuristics: Fault-Tolerant Inference for MoE LLMs on Heterogeneous Edge Devices

> **Source:** *Fault-Tolerant Inference for MoE LLMs on Heterogeneous, Resource-Constrained Edge Devices*, project report, October 2025.
>
> **Purpose:** Convert the report's stated mechanisms, decision rules, and implementation guidance into an implementation-oriented checklist. This document records what the report says; it does not add external design decisions or claim that proposed mechanisms have already been implemented or validated.

## 1. Scope and interpretation

The proposed system keeps the original MoE model and router unchanged. It optimizes expert placement and movement across devices and storage tiers. Expert aggregation/compression is explicitly out of scope for the proposed extension. [Report, §§3.1, 3.4, pp. 13–14]

The report combines:
- mechanisms inherited or adapted from CoMoE;
- new mechanisms for CPU-only devices and disk-backed storage;
- proposed fault-tolerant coordination and replication;
- implementation steps and open design decisions.

The labels below distinguish these categories. Equations are transcribed in report notation where practical. The report does not specify concrete numeric values for most thresholds or weights.

## 2. Core system invariants

1. **Keep the model fixed.** Do not merge, prune, or otherwise modify experts as part of this project's placement optimization. [§3.1, p. 13]
2. **Use a fixed execution type per device.** A device with a usable GPU executes expert computation on its GPU; a CPU-only device executes expert computation on its CPU. The execution type is not a runtime optimization variable. [§3.2, p. 13]
3. **Do not execute experts from disk.** Disk is a capacity tier only. Experts must be loaded into CPU memory, and then into GPU memory when the destination device executes on a GPU. [§3.3, pp. 13–14]
4. **Respect each tier's memory capacity.** Placement must satisfy the memory capacity of every storage tier on every device. [§3.4, p. 14]
5. **Optimize placement and movement, not aggregation.** The proposed optimization variable is expert placement across tiers and devices. [§3.4, p. 14]
6. **Account for computation and communication separately.** End-to-end latency includes both expert computation and the transfers required to make activated experts available. [§3.5, pp. 14–15]

## 3. Device resource perception

### 3.1 Track a time-varying resource state

For each device \(d_i\), maintain a resource state:

\[
R_i(t)=\{C_i(t),M_i(t),B_i(t)\}
\]

The report describes these dimensions as:
- **Compute:** CPU/GPU compute capacity and utilization.
- **Memory:** total and used memory.
- **Bandwidth/network:** internal and network bandwidth and latency.

Treat these as time-varying measurements, not fixed device specifications. [CoMoE background, §2.2.1–2.2.2, pp. 7–9; proposed model, §3.2, p. 13]

### 3.2 Smooth noisy measurements with EWMA

For each scalar resource indicator \(r(t)\), maintain an exponentially weighted moving average:

\[
\hat r(t)=\alpha r(t)+(1-\alpha)\hat r(t-1),\qquad 0<\alpha<1
\]

Use the smoothed value in resource-aware decisions rather than reacting directly to every raw measurement. The report does not prescribe a value for \(\alpha\). [§2.2.2, pp. 8–9]

### 3.3 Estimate resource stability

The report defines a stability indicator using short-term volatility relative to the smoothed resource level:

\[
S_r(t)=1-\frac{\sigma_r(t)}{\hat r(t)}
\]

with rolling-window volatility:

\[
\sigma_r(t)=
\sqrt{\frac{1}{w}\sum_{i=t-w+1}^{t}(r(i)-\hat r(t))^2}
\]

Interpretation in the report:
- A value near 1 indicates a stable resource.
- A falling value indicates sharp fluctuation.
- Make the framework more conservative as stability falls.

The report does not specify the window length \(w\), handling for a zero or near-zero denominator, or clipping behavior. [§2.2.2, p. 9]

### 3.4 Quantify device heterogeneity

For devices \(d_i\) and \(d_j\), the report defines a weighted normalized resource difference:

\[
D_{ij}=\sum_{r\in R}w_r
\frac{|r_{d_i}-r_{d_j}|}{\max(r_{d_i},r_{d_j})}
\]

and a device-level heterogeneity score:

\[
H_{d_i}=\frac{1}{N-1}\sum_{j\ne i}D_{ij}
\]

The dimension weights \(w_r\) sum to 1. A higher score means the device differs more from its peers. The report does not specify the weights or zero-denominator handling. [§2.2.2, p. 9]

### 3.5 Predict performance from resource state

The CoMoE background models computation latency as a polynomial regression over resource-state and model-configuration features, then refines coefficients online using gradient descent on prediction error. Communication latency and memory usage are predicted similarly. [§2.2.2, p. 9]

**Implementation heuristic:** maintain estimates for compute latency, communication latency, and memory use; update the estimates as observed outcomes arrive so the model can track changing device behavior.

The report does not provide a concrete feature list, polynomial degree, learning rate, or prediction-error loss configuration.

## 4. Expert placement and storage-tier heuristics

### 4.1 Use device-specific storage ladders

**GPU-capable device:**

1. GPU workspace — experts currently in use.
2. GPU cache — experts predicted to be needed soon.
3. CPU memory — backing store.
4. Disk — overflow/capacity tier.

**CPU-only device:**

1. CPU memory — storage and execution tier.
2. Disk — overflow/capacity tier.

Disk is never an execution tier. [§3.3, pp. 13–14; Figure 5, p. 14]

### 4.2 Move experts only along supported tier paths

- GPU-capable device: GPU ↔ CPU memory; CPU memory ↔ disk.
- CPU-only device: CPU memory ↔ disk.
- Disk-to-GPU retrieval on a GPU-capable device is a two-stage path: disk → CPU memory → GPU.
- Disk-to-CPU retrieval on a CPU-only device is a one-stage path: disk → CPU memory.

These paths determine the loading and offloading costs. [§3.3–3.5, pp. 13–15]

### 4.3 Model loading cost by current tier and execution type

For an expert \(e\) of size \(s_e\), the report's loading-cost cases are:

| Device type | Expert's current tier | Loading cost |
|---|---|---|
| GPU | GPU | \(0\) |
| GPU | CPU memory | \(s_e/b_i^{cpu\text{-}gpu}(t)\) |
| GPU | Disk | \(s_e/b_i^{disk\text{-}cpu}(t)+s_e/b_i^{cpu\text{-}gpu}(t)\) |
| CPU-only | CPU memory | \(0\) |
| CPU-only | Disk | \(s_e/b_i^{disk\text{-}cpu}(t)\) |

For a token \(x_t\) at layer \(l\), sum the loading costs of all activated experts:

\[
L_{comm}(x_t,l)=\sum_{e\in A(x_t,l)}L_{load}(e,t)
\]

Include this communication cost alongside computation latency when evaluating placement. [§3.5, pp. 14–15]

### 4.4 Make placement a constrained latency optimization

The proposed optimization is:

\[
\min_S L_{total}(X,\mathcal M,S,\mathcal D)
\]

subject to the memory capacity of every tier, with the model held fixed and each device's execution type fixed. The decision variable is expert placement \(S\), including where experts reside and when they move. [§3.4, p. 14]

## 5. Offloading, prefetching, and cache management

These mechanisms adapt CoMoE's expert offloading approach to the report's GPU and CPU-only storage hierarchies. [§3.6, p. 15]

### 5.1 Assign offloading priority using popularity and retrieval cost

The report gives the priority score:

\[
P_{offload}(e,t)=
\gamma f_{act}(e,t)+(1-\gamma)\frac{s_e}{L_{comm}(e,t)}
\]

where:
- \(f_{act}(e,t)\) is expert activation frequency;
- \(s_e\) is expert size;
- \(L_{comm}(e,t)\) is its communication/retrieval cost;
- \(\gamma\) controls the trade-off.

Use the score to balance activation frequency against retrieval cost when deciding which experts to offload. The report does not specify the direction/order in which this score is consumed by an eviction or placement algorithm, nor a value for \(\gamma\). [§3.6, p. 15]

### 5.2 Predict next-layer expert activations

CoMoE's described predictor is a lightweight two-layer MLP. Its input combines:
- experts selected in the current layer;
- the current token's hidden state;
- context such as positional embeddings;
- activation history.

It predicts probabilities for experts in the next layer. [§2.2.5, pp. 11–12]

**Implementation heuristic:** use predicted activation probabilities to decide which experts to prefetch before the next layer needs them. The report does not give a full training procedure or deployment specification for the predictor.

### 5.3 Use an adaptive prefetch threshold

The proposed GPU-device threshold is:

\[
\theta_{pre}(t)=
\theta_{base}S_b(t)
\left(1+\delta\frac{M^{gpu}_{avail}(t)}{M^{gpu}}\right)
\]

The report describes the threshold as responding to bandwidth stability and spare GPU memory:
- more available GPU memory permits more aggressive prefetching;
- less available memory reduces prefetching;
- lower bandwidth stability makes the system more conservative.

Prefetch an expert when its predicted activation probability clears the applicable threshold. [CoMoE background, §2.2.3, pp. 9–10; proposed adaptation, §3.6, p. 15]

For CPU-only devices, the GPU-memory term is absent. Prefetching means moving an expert from disk into CPU memory under a CPU-side threshold \(\theta_{pre}^{cpu}\). The report names this threshold but does not provide a complete explicit formula for it. [§3.6, p. 15]

### 5.4 Evict using predicted inactivity, recency, and importance

The report's cache eviction score is:

\[
S_{evict}(e_i)=
(1-\delta)(1-p(e_i))+
\delta\left(
\lambda\frac{1}{f_{recent}(e_i)}
+(1-\lambda)\frac{1}{I(e_i)}
\right)
\]

where:
- \(p(e_i)\) is predicted activation probability;
- \(f_{recent}(e_i)\) represents recency/frequency information;
- \(I(e_i)\) is expert importance;
- \(\delta\) and \(\lambda\) control the combination.

Experts with the highest eviction score are moved back to CPU memory first (for GPU cache eviction). The report does not define the exact recency window or importance calculation in the proposed section. [CoMoE background, §2.2.5, p. 12; proposed adaptation, §3.6, p. 15]

### 5.5 Correct prediction errors

CoMoE's described correction pass uses popularity, importance, and similarity:
- high-priority mispredictions trigger immediate cache replacement;
- lower-priority mispredictions can be substituted with a similar expert already resident on the GPU, avoiding a transfer.

This is described in the CoMoE literature review; the proposed work says it adapts CoMoE's offloading mechanisms, but does not separately specify a complete CPU-only version of this correction pass. [§2.2.5, p. 12; §3.6, p. 15]

### 5.6 Adapt at two timescales

The report retains a two-level adaptation pattern:
- **Coarse timescale:** adjust baseline storage allocation (how many experts are held in GPU memory, CPU memory, and disk) in response to sustained resource changes.
- **Fine timescale:** perform per-token expert prediction, prefetching, cache replacement, and scheduling on both GPU-capable and CPU-only devices.

Unlike CoMoE, the proposed system does not switch between compressed model variants because the model remains fixed. [§3.7, p. 15]

## 6. Fault-tolerant coordination heuristics

The following mechanisms are proposed in the report; the current prototype does not yet implement them. [§3.8, p. 15; §4.2–4.3, pp. 17–18]

### 6.1 Avoid a permanently designated coordinator

- Allow any device to act as coordinator.
- Do not make system operation depend permanently on one pre-assigned orchestrator. [§3.8, p. 15]

### 6.2 Detect coordinator failure with heartbeats

- Have devices exchange heartbeats.
- Treat a coordinator that stops responding as a failure condition that triggers replacement election.

The report does not specify heartbeat intervals, timeout values, or failure-detection tuning. [§3.8, p. 15]

### 6.3 Elect a replacement by majority vote

- Use a majority vote among surviving devices to elect a replacement coordinator.
- The report identifies Raft as the consensus-protocol model for this coordination approach.

The report does not specify a complete Raft configuration, membership-change procedure, quorum behavior for particular cluster sizes, or split-brain handling. [§3.8, p. 15]

### 6.4 Commit generated tokens before emitting them

- The coordinator commits each generated token before emitting it to the user.
- Maintain token history as the replicated-log analogue.
- After coordinator failover, do not regenerate text that has already been emitted.
- Recompute only the in-flight token.

This is the report's stated mechanism for avoiding duplicate/repeated output across coordinator failover. [§3.8, p. 15]

### 6.5 Replicate expert groups to a neighbor

- Each device backs up a neighboring device's expert group.
- If a device is lost, reroute requests for its expert group to the backup holder.
- The proposed approach aims to absorb a device loss through backup routing rather than re-sharding the model.

The report does not fully specify replica consistency, backup placement selection, or recovery when both a primary and its backup are unavailable. [§3.8, p. 15; §4.3, pp. 18]

### 6.6 Replicate main weights for coordinator replacement

The future-work roadmap says every device should hold a standby copy of the main weights. This supports coordinator replacement without depending on state held only by the failed main node. [§4.3, p. 18]

## 7. Implementation roadmap extracted from the report

The report presents the following order as the roadmap from the existing prototype to the proposed system.

### Phase 1 — Heterogeneous devices

- Move from containers on one machine to separate machines.
- Address devices by host and port.
- Copy each shard to its device.
- Add timeouts and retries.
- Introduce a real per-device CPU/GPU execution type.
- Allow CPU-only devices to join the inference pool.

**Report status:** future work; not present in the current prototype. [§4.2–4.3, pp. 17–18]

### Phase 2 — Multi-tier offloading

- Implement expert placement state.
- Add the disk storage tier.
- Implement the disk-aware loading-cost model.
- Add adaptive prefetching and eviction.
- Support the different storage paths for GPU-capable and CPU-only devices.
- Replace the current assumption that all experts remain resident.

**Report status:** identified as the largest remaining piece of work; these mechanisms are not exercised by the current prototype. [§4.2–4.3, p. 18]

### Phase 3 — Fault tolerance

- Give every device a standby copy of the main weights.
- Add heartbeat exchange and timeout-based failure detection.
- Add majority-based coordinator election.
- Add commit-before-emit token logging.
- Back up each device's expert group on a neighbor.
- Reroute requests to the backup holder after device loss.

**Report status:** future work; the current main node remains a single point of failure. [§4.2–4.3, pp. 17–18]

### Phase 4 — Intra-device expert parallelism

- Run the selected experts in parallel within each container/device rather than sequentially.

The report describes this as a modest change to the existing code path. [§4.3, p. 18]

## 8. Explicitly unresolved design decisions

The report leaves these choices open; they should not be treated as settled design requirements.

1. **Lost expert group handling:** decide whether a lost expert group is replicated, masked from the router, or migrated to a surviving device. [§4.3, p. 18]
2. **Attention-cache failover:** decide whether the attention cache is rebuilt after failover or replicated per token. [§4.3, p. 18]
3. **Resource-dependent choices:** the report says both decisions depend on the memory available on the target devices. [§4.3, p. 18]

Other implementation parameters are also not numerically fixed in the report, including EWMA coefficient, stability window, resource-dimension weights, predictor configuration, prefetch thresholds, eviction weights, heartbeat interval, and failure timeout.

## 9. Related CoMoE heuristics described in the report but outside the proposed implementation scope

This section preserves relevant heuristics from the literature review while separating them from the project's stated implementation scope. The proposed work explicitly excludes expert aggregation and keeps the original model unchanged. [§3.1, p. 13]

### 9.1 Expert-load balancing during MoE training

The MoE background describes two auxiliary losses intended to discourage routing collapse:
- **Importance loss:** penalize unequal batchwise sums of gate weights across experts using a squared coefficient of variation.
- **Load loss:** apply a similar penalty to a smooth estimate of the fraction of examples routed to each expert.

These are training-time routing-balancing mechanisms, not placement or fault-tolerance mechanisms in the proposed extension. [§2.1, pp. 5–7; reference [1]]

### 9.2 CoMoE resource-aware aggregation

The CoMoE review describes:
- choosing a target expert count based on available GPU memory and memory stability;
- measuring expert similarity using parameter-space cosine similarity and functional-space output-distribution similarity;
- identifying principal experts using activation frequency;
- assigning other experts to their most similar principal expert;
- merging groups using activation-frequency-weighted averages;
- using activation entropy to retain more experts in layers with more distributed activation;
- selecting a precomputed model variant that fits available memory;
- switching variants only when expected performance gain exceeds switching cost and the resource state has remained stable long enough.

These mechanisms belong to CoMoE's aggregation path. The proposed project removes aggregation from its optimization and does not plan to modify the model. [§2.2.3–2.2.4, pp. 9–11; §3.1, p. 13]

### 9.3 CoMoE-specific prediction and caching details

The report's CoMoE review also describes:
- a two-layer MLP for next-layer expert activation prediction;
- a GPU workspace, GPU cache, and CPU backing store;
- adaptive prefetching based on predicted probability and free GPU memory;
- eviction based on predicted inactivity, recency, and importance;
- a correction pass using popularity, importance, and expert similarity.

The proposed system adapts the offloading/prefetching/eviction family to CPU-only and disk-backed tiers. It does not state that every CoMoE detail is carried over unchanged. [§2.2.5, pp. 11–12; §3.3, §3.6, pp. 13, 15]

## 10. Existing prototype facts and constraints

These are implementation facts reported about the baseline, not heuristics to assume are already implemented in the proposed system:

- The prototype distributes IBM Granite 3.1 1B-A400M across five Docker containers on one machine.
- One main container holds embeddings, output matrices, attention blocks, layer norms, routers, and the attention cache.
- Four expert containers hold contiguous expert-index groups from every layer.
- The main node routes top-8 experts and sends requests to expert containers in parallel.
- Experts within each container currently execute sequentially.
- The prototype has no heterogeneous device pool, no CPU-only execution distinction, no offloading, no disk tier, no timeouts/retries, and no fault-tolerant coordination.
- The main node is a single point of failure.

[§4.1–4.2, pp. 17–18]

## 11. Evaluation measures named in the report

The report says the proposed approach is to be evaluated using:
- inference latency;
- throughput;
- resource utilization;
- recovery time;
- successful inference completion under varying hardware configurations and failure conditions.

These are evaluation dimensions named by the report, not reported results for the proposed fault-tolerant extension. [Abstract, p. 3; §1, p. 4]

## 12. Source references

In-text citations above use the report's section numbers and printed page numbers. The report's bibliography is reproduced here for the works it cites:

1. N. Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.” https://arxiv.org/abs/1701.06538
2. M. Li et al., “CoMoE: Collaborative Optimization of Expert Aggregation and Offloading for MoE-based LLMs at Edge.” https://arxiv.org/abs/2508.09208
3. O. Sanseviero et al., “Mixture of Experts Explained.” https://huggingface.co/blog/moe
4. W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.” https://arxiv.org/abs/2101.03961
5. L. Xue et al., “MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache.” https://arxiv.org/abs/2401.14361
6. D. Ongaro and J. Ousterhout, “In Search of an Understandable Consensus Algorithm,” USENIX ATC 2014. https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro
7. D. Lepikhin et al., “GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.” https://arxiv.org/abs/2006.16668
8. S. Rajbhandari et al., “DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.” https://arxiv.org/abs/2201.05596

---

**Extraction boundary:** This document is an implementation-oriented extraction of the uploaded project report. It preserves distinctions between background literature, the proposed design, the current prototype, and future work. It does not resolve unspecified parameters or open design decisions.

<script lang="ts">
	import type { NodeInfo } from '$lib/types';
	import { fmtBytes, fmtMbps, fmtMs, fmtPct, fmtRatio } from '$lib/format';

	interface Props {
		node: NodeInfo;
	}

	let { node }: Props = $props();
	let p = $derived(node.profile);
	let r = $derived(node.resources);
	let res = $derived(node.residency);
</script>

<div class="card detail">
	<div class="head">
		<span class="dot {node.reachable ? 'ok' : 'down'}"></span>
		<b>{node.name}</b>
		<span class="badge {node.execution === 'gpu' ? 'gpu' : 'cpu'}">{node.execution ?? '?'}</span>
		<span class="muted">{node.role}</span>
		<span class="spacer"></span>
		<span class="muted">{node.rttMs != null ? `${node.rttMs.toFixed(0)}ms` : '—'}</span>
	</div>

	<h4>resources · stability</h4>
	<div class="kv">
		<span class="k">CPU util</span>
		<span>{fmtPct(r?.cpu_util)} <span class="muted">stab {fmtRatio(r?.cpu_util_stability)}</span></span>
	</div>
	<div class="kv">
		<span class="k">Memory avail</span>
		<span>{fmtBytes(r?.mem_available_bytes)} <span class="muted">stab {fmtRatio(r?.mem_stability)}</span></span>
	</div>
	<div class="kv">
		<span class="k">GPU free</span>
		<span>{fmtBytes(r?.gpu_free_bytes)}</span>
	</div>

	<h4>compute</h4>
	<div class="kv"><span class="k">CPU cores</span><span>{p?.cpu_cores ?? '—'}</span></div>
	<div class="kv"><span class="k">CPU</span><span>{p ? `${p.cpu_gflops} GFLOPS` : '—'}</span></div>
	<div class="kv"><span class="k">GPU</span><span>{p?.gpu_tflops != null ? `${p.gpu_tflops} TFLOPS` : '—'}</span></div>

	<h4>expert latency (cost model)</h4>
	<div class="kv"><span class="k">fixed</span><span>{fmtMs(p?.expert_fixed_ms)}</span></div>
	<div class="kv"><span class="k">per expert</span><span>{fmtMs(p?.expert_per_ms)}</span></div>

	<h4>memory & bandwidth</h4>
	<div class="kv"><span class="k">RAM</span><span>{fmtBytes(p?.ram_bytes)}</span></div>
	<div class="kv"><span class="k">VRAM</span><span>{fmtBytes(p?.vram_bytes)}</span></div>
	<div class="kv"><span class="k">disk read</span><span>{fmtMbps(p?.disk_read_mbps)}</span></div>
	<div class="kv"><span class="k">disk write</span><span>{fmtMbps(p?.disk_write_mbps)}</span></div>

	{#if res}
		<h4>expert residency · eviction</h4>
		<div class="kv"><span class="k">eviction budget</span><span>{fmtBytes(res.budget_bytes)}</span></div>
		<div class="kv"><span class="k">resident</span><span>{fmtBytes(res.used_bytes)}</span></div>
		<div class="kv">
			<span class="k">free</span>
			<span>{fmtBytes(res.free_bytes)} <span class="muted">({fmtPct(res.free_ratio)})</span></span>
		</div>
		<div class="kv">
			<span class="k">θ_pre threshold</span>
			<span>{res.prefetch_threshold != null ? res.prefetch_threshold.toFixed(4) : '—'}</span>
		</div>
		<div class="kv"><span class="k">γ · freq weight</span><span>{res.config.gamma}</span></div>
		<div class="kv"><span class="k">δ · recency</span><span>{res.config.delta}</span></div>
		<div class="kv"><span class="k">λ · importance</span><span>{res.config.lambda}</span></div>
		<div class="kv"><span class="k">θ · prefetch base</span><span>{res.config.theta_base}</span></div>
		<div class="kv"><span class="k">θ · memory</span><span>{res.config.theta_mem}</span></div>
		<div class="kv"><span class="k">decay</span><span>{res.config.decay}</span></div>
	{/if}
</div>

<style>
	.detail {
		display: flex;
		flex-direction: column;
	}

	.head {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}

	.spacer {
		flex: 1;
	}

	h4 {
		margin: 12px 0 4px;
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--muted);
	}

	h4:first-of-type {
		margin-top: 0;
	}
</style>

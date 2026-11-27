<script lang="ts">
	import type { ClusterSnapshot } from '$lib/types';
	import { fmtRanges } from '$lib/format';

	interface Props {
		snapshot: ClusterSnapshot | null;
	}

	let { snapshot }: Props = $props();

	let byNode = $derived.by(() => {
		const groups: Record<string, number[]> = {};
		const assignment = snapshot?.routing?.assignment ?? {};
		for (const [e, entry] of Object.entries(assignment)) {
			(groups[entry.node] ??= []).push(Number(e));
		}
		return Object.entries(groups)
			.map(([node, experts]) => ({ node, experts: experts.sort((a, b) => a - b) }))
			.sort((a, b) => a.node.localeCompare(b.node));
	});

	let topExperts = $derived.by(() => {
		const counts = snapshot?.experts?.counts ?? [];
		return counts
			.map((c, e) => ({ e, c }))
			.filter((t) => t.c > 0)
			.sort((a, b) => b.c - a.c)
			.slice(0, 8);
	});

	let maxCount = $derived(Math.max(1, ...topExperts.map((t) => t.c)));
</script>

<div class="expert-panel scroll">
	<section class="card">
		<h4>routing · experts per node</h4>
		{#each byNode as g (g.node)}
			<div class="node-row">
				<span class="node-name">{g.node}</span>
				<span class="node-count">{g.experts.length} experts</span>
			</div>
			<div class="node-ranges muted">{fmtRanges(g.experts)}</div>
		{/each}
	</section>

	<section class="card">
		<h4>top activations</h4>
		{#if topExperts.length === 0}
			<p class="muted">no activations yet</p>
		{:else}
			{#each topExperts as t (t.e)}
				<div class="expert-row">
					<span class="e-name">e{t.e}</span>
					<span class="e-bar"><span class="e-fill" style="width:{(t.c / maxCount) * 100}%"></span></span>
					<span class="e-count">{t.c}</span>
				</div>
			{/each}
		{/if}
	</section>

	<section class="card">
		<h4>predictor</h4>
		{#if snapshot?.predictor && snapshot.predictor.samples > 0}
			<div class="kv"><span class="k">kind</span><span>{snapshot.predictor.kind}</span></div>
			<div class="kv">
				<span class="k">top-1</span>
				<span>{((snapshot.predictor.top1_hits / snapshot.predictor.samples) * 100).toFixed(1)}%</span>
			</div>
			<div class="kv">
				<span class="k">top-3</span>
				<span>{((snapshot.predictor.top3_hits / snapshot.predictor.samples) * 100).toFixed(1)}%</span>
			</div>
		{:else}
			<p class="muted">no predictor trained yet</p>
		{/if}
	</section>
</div>

<style>
	.expert-panel {
		width: 300px;
		flex-shrink: 0;
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding-left: 2px;
	}

	h4 {
		margin: 0 0 8px;
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--muted);
	}

	.node-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	.node-name {
		font-weight: 600;
	}

	.node-count {
		color: var(--muted);
		font-size: 12px;
	}

	.node-ranges {
		font-size: 12px;
		margin: 2px 0 8px;
		word-break: break-word;
	}

	.expert-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-variant-numeric: tabular-nums;
	}

	.e-name {
		width: 28px;
		color: var(--muted);
	}

	.e-bar {
		flex: 1;
		height: 6px;
		background: var(--panel-2);
		border-radius: 3px;
		overflow: hidden;
	}

	.e-fill {
		display: block;
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}

	.e-count {
		width: 40px;
		text-align: right;
	}

	.kv {
		display: flex;
		justify-content: space-between;
		font-variant-numeric: tabular-nums;
	}

	.kv .k {
		color: var(--muted);
	}

	p {
		margin: 0;
	}
</style>

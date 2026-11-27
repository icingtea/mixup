<script lang="ts">
	import type { ClusterSnapshot } from '$lib/types';

	interface Props {
		snapshot: ClusterSnapshot | null;
	}

	let { snapshot }: Props = $props();

	let reachable = $derived(snapshot ? snapshot.nodes.filter((n) => n.reachable).length : 0);
	let tokens = $derived(snapshot ? snapshot.nodes.reduce((a, n) => a + n.totalTokens, 0) : 0);
	let experts = $derived(snapshot ? new Set(snapshot.nodes.flatMap((n) => n.ownedExperts)).size : 0);
</script>

<div class="summary">
	<div class="card">
		<h3>cluster</h3>
		<p>leader: <b>{snapshot?.leader ?? 'none'}</b></p>
		<p>nodes: {snapshot?.nodes.length ?? 0} (<span class="green">{reachable}</span> reachable)</p>
		<p>experts owned: {experts}</p>
		<p>tokens generated: {tokens}</p>
	</div>

	<div class="card">
		<h3>predictor</h3>
		<p>kind={snapshot?.predictor?.kind ?? 'none'} · samples={snapshot?.predictor?.samples ?? 0}</p>
		<p>top-1 {snapshot?.predictor?.top1_hits ?? 0}/{snapshot?.predictor?.samples ?? 0}</p>
	</div>

	<div class="card">
		<h3>phases</h3>
		<p>0–6 done · 7 (deploy) deferred</p>
	</div>
</div>

<style>
	.summary {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
		gap: 14px;
	}

	h3 {
		margin: 0 0 8px;
		font-size: 13px;
		color: var(--muted);
		font-weight: 600;
	}

	p {
		margin: 4px 0;
	}
</style>

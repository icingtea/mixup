<script lang="ts">
	import type { PredictorInfo } from '$lib/types';

	interface Props {
		predictor: PredictorInfo | null;
	}

	let { predictor }: Props = $props();
</script>

{#if !predictor || predictor.kind === 'none' || predictor.samples === 0}
	<p class="muted">no predictor trained yet — generate a few tokens on the leader.</p>
{:else}
	<div class="card">
		<h3>{predictor.kind} predictor · version {predictor.version}</h3>
		<p>samples: <b>{predictor.samples}</b></p>
		<p>
			top-1: {predictor.top1_hits}/{predictor.samples} =
			<span class="green">{((predictor.top1_hits / predictor.samples) * 100).toFixed(1)}%</span>
			<span class="muted"> (misses {predictor.samples - predictor.top1_hits})</span>
		</p>
		<p>
			top-3: {predictor.top3_hits}/{predictor.samples} =
			<span class="green">{((predictor.top3_hits / predictor.samples) * 100).toFixed(1)}%</span>
			<span class="muted"> (misses {predictor.samples - predictor.top3_hits})</span>
		</p>
	</div>
{/if}

<style>
	h3 {
		margin: 0 0 8px;
		font-size: 13px;
		color: var(--muted);
		font-weight: 600;
	}
</style>

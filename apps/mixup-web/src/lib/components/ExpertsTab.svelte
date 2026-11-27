<script lang="ts">
	import type { ExpertStats } from '$lib/types';
	import BarChart from './BarChart.svelte';

	interface Props {
		experts: ExpertStats | null;
	}

	let { experts }: Props = $props();

	let ranked = $derived.by(() => {
		const counts = experts?.counts ?? [];
		return counts
			.map((c, e) => ({ e, c }))
			.sort((a, b) => b.c - a.c)
			.slice(0, 16);
	});

	let total = $derived((experts?.counts ?? []).reduce((a, c) => a + c, 0));
</script>

{#if !experts || experts.counts.length === 0}
	<p class="muted">no expert activations yet — generate a few tokens first.</p>
{:else}
	<div class="card">
		<h3>top experts by activation</h3>
		<BarChart
			labels={ranked.map((t) => `e${t.e}`)}
			data={ranked.map((t) => t.c)}
			color="#58a6ff"
			xLabel="expert"
			yLabel="activations"
		/>
	</div>

	<div class="card">
		<table>
			<thead>
				<tr>
					<th>expert</th>
					<th>activations</th>
					<th>share</th>
				</tr>
			</thead>
			<tbody>
				{#each ranked as t (t.e)}
					<tr>
						<td>expert {t.e}</td>
						<td>{t.c}</td>
						<td>{total > 0 ? ((t.c / total) * 100).toFixed(1) : 0}%</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}

<style>
	h3 {
		margin: 0 0 8px;
		font-size: 13px;
		color: var(--muted);
		font-weight: 600;
	}

	.card + .card {
		margin-top: 14px;
	}
</style>

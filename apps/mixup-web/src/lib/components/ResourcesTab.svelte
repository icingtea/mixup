<script lang="ts">
	import type { ClusterSnapshot, HistorySeries } from '$lib/types';
	import LineChart from './LineChart.svelte';

	interface Props {
		snapshot: ClusterSnapshot | null;
	}

	let { snapshot }: Props = $props();

	const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#7ee787'];

	function nodeList() {
		return snapshot?.nodes ?? [];
	}

	function series(name: string): HistorySeries {
		return snapshot?.history[name] ?? { t: [], cpu: [], mem: [], gpu: [], rtt: [] };
	}

	function labels(): string[] {
		const first = nodeList()[0];
		const t = first ? series(first.name).t : [];
		if (!t.length) return [];
		const t0 = t[0];
		return t.map((v) => `${((v - t0) / 1000).toFixed(0)}s`);
	}

	function datasets(field: (h: HistorySeries) => number[]) {
		return nodeList().map((n, i) => ({
			label: n.name,
			data: field(series(n.name)),
			color: COLORS[i % COLORS.length]
		}));
	}
</script>

<div class="resources">
	<div class="card">
		<h3>CPU utilisation (%)</h3>
		<LineChart
			labels={labels()}
			datasets={datasets((h) => h.cpu.map((v) => v * 100))}
			yLabel="%"
			min={0}
			max={100}
		/>
	</div>

	<div class="card">
		<h3>Memory available (GB)</h3>
		<LineChart labels={labels()} datasets={datasets((h) => h.mem.map((v) => v / 1e9))} yLabel="GB" />
	</div>

	<div class="card">
		<h3>GPU memory free (GB)</h3>
		<LineChart
			labels={labels()}
			datasets={datasets((h) => h.gpu.map((v) => (v > 0 ? v / 1e9 : 0)))}
			yLabel="GB"
		/>
	</div>

	<div class="card">
		<h3>Control-plane RTT (ms)</h3>
		<LineChart labels={labels()} datasets={datasets((h) => h.rtt)} yLabel="ms" min={0} />
	</div>
</div>

<style>
	.resources {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 14px;
	}

	h3 {
		margin: 0 0 8px;
		font-size: 13px;
		color: var(--muted);
		font-weight: 600;
	}

	@media (max-width: 900px) {
		.resources {
			grid-template-columns: 1fr;
		}
	}
</style>

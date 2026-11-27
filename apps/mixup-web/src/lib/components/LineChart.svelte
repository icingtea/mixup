<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Chart,
		LineController,
		LineElement,
		PointElement,
		LinearScale,
		CategoryScale,
		Tooltip,
		Legend,
		Filler
	} from 'chart.js';

	Chart.register(
		LineController,
		LineElement,
		PointElement,
		LinearScale,
		CategoryScale,
		Tooltip,
		Legend,
		Filler
	);

	interface Dataset {
		label: string;
		data: number[];
		color: string;
		fill?: boolean;
	}

	interface Props {
		labels: string[];
		datasets: Dataset[];
		yLabel?: string;
		min?: number;
		max?: number;
	}

	let { labels, datasets, yLabel, min, max }: Props = $props();

	let canvas: HTMLCanvasElement;
	let chart: Chart | undefined;

	onMount(() => {
		chart = new Chart(canvas, {
			type: 'line',
			data: {
				labels,
				datasets: datasets.map((d) => ({
					label: d.label,
					data: d.data,
					borderColor: d.color,
					backgroundColor: d.color + '22',
					tension: 0.3,
					pointRadius: 0,
					borderWidth: 1.5,
					fill: d.fill ?? false
				}))
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				interaction: { mode: 'nearest', intersect: false },
				scales: {
					x: {
						ticks: { color: '#71717a', maxTicksLimit: 8, maxRotation: 0 },
						grid: { color: '#f0f0f1' }
					},
					y: {
						title: yLabel ? { display: true, text: yLabel, color: '#71717a' } : undefined,
						min,
						max,
						ticks: { color: '#71717a' },
						grid: { color: '#f0f0f1' }
					}
				},
				plugins: { legend: { labels: { color: '#18181b' } } }
			}
		});
		return () => chart?.destroy();
	});

	$effect(() => {
		if (!chart) return;
		chart.data.labels = labels;
		chart.data.datasets.forEach((d, i) => {
			if (datasets[i]) d.data = datasets[i].data;
		});
		chart.update('none');
	});
</script>

<div class="chart-box">
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.chart-box {
		position: relative;
		height: 150px;
	}
</style>

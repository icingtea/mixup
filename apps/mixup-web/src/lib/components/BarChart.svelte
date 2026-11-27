<script lang="ts">
	import { onMount } from 'svelte';
	import { Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend } from 'chart.js';

	Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend);

	interface Props {
		labels: string[];
		data: number[];
		color: string;
		xLabel?: string;
		yLabel?: string;
	}

	let { labels, data, color, xLabel, yLabel }: Props = $props();

	let canvas: HTMLCanvasElement;
	let chart: Chart | undefined;

	onMount(() => {
		chart = new Chart(canvas, {
			type: 'bar',
			data: { labels, datasets: [{ label: yLabel ?? '', data, backgroundColor: color, borderRadius: 2 }] },
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: false,
				scales: {
					x: {
						ticks: { color: '#71717a' },
						grid: { display: false },
						title: xLabel ? { display: true, text: xLabel, color: '#71717a' } : undefined
					},
					y: {
						beginAtZero: true,
						ticks: { color: '#71717a' },
						grid: { color: '#f0f0f1' }
					}
				},
				plugins: { legend: { display: false } }
			}
		});
		return () => chart?.destroy();
	});

	$effect(() => {
		if (!chart) return;
		chart.data.labels = labels;
		chart.data.datasets[0].data = data;
		chart.update('none');
	});
</script>

<div class="chart-box">
	<canvas bind:this={canvas}></canvas>
</div>

<style>
	.chart-box {
		position: relative;
		height: 180px;
	}
</style>

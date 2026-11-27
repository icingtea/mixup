<script lang="ts">
	import type { NodeInfo } from '$lib/types';

	interface Props {
		nodes: NodeInfo[];
		leader: string | null;
		onToggle: (name: string, action: 'stop' | 'start') => void;
	}

	let { nodes, leader, onToggle }: Props = $props();
</script>

<div class="sidebar scroll">
	<div class="sidebar-title">nodes</div>
	{#each nodes as n (n.name)}
		<div class="node card">
			<div class="row">
				<span class="dot {n.reachable ? 'ok' : 'down'}"></span>
				<span class="name">{n.name}</span>
				{#if n.name === leader}<span class="badge leader">LEADER</span>{/if}
			</div>
			<div class="row muted">
				<span>{n.role}</span>
				<span class="badge {n.execution === 'gpu' ? 'gpu' : 'cpu'}">{n.execution ?? '?'}</span>
				<span>{n.ownedExperts.length} experts</span>
			</div>
			<div class="row muted">
				<span>cpu {((n.resources?.cpu_util ?? 0) * 100).toFixed(1)}%</span>
				{#if n.rttMs != null}<span>rtt {n.rttMs.toFixed(0)}ms</span>{/if}
			</div>
			<div class="row">
				<button class="danger" onclick={() => onToggle(n.name, 'stop')}>kill</button>
				<button onclick={() => onToggle(n.name, 'start')}>start</button>
			</div>
		</div>
	{/each}
</div>

<style>
	.sidebar {
		width: 260px;
		flex-shrink: 0;
		border-right: 1px solid var(--border);
		background: var(--panel);
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.sidebar-title {
		color: var(--muted);
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.node {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.row {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.name {
		font-weight: 600;
	}

	.row button {
		flex: 1;
		padding: 3px 6px;
		font-size: 12px;
	}
</style>

<script lang="ts">
	import { onMount } from 'svelte';
	import type { ClusterSnapshot, GenerateResult } from '$lib/types';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import ChatTab from '$lib/components/ChatTab.svelte';
	import NodesTab from '$lib/components/NodesTab.svelte';
	import ResourcesTab from '$lib/components/MetricsTab.svelte';
	import ExpertsTab from '$lib/components/ExpertsTab.svelte';
	import PredictorTab from '$lib/components/PredictorTab.svelte';
	import SummaryTab from '$lib/components/SummaryTab.svelte';
	import LocalTab from '$lib/components/LocalTab.svelte';

	const POLL = 1000;
	const TABS = [
		{ id: 'chat', label: 'Chat' },
		{ id: 'nodes', label: 'Nodes' },
		{ id: 'resources', label: 'Metrics' },
		{ id: 'experts', label: 'Experts' },
		{ id: 'predictor', label: 'Predictor' },
		{ id: 'summary', label: 'Summary' },
		{ id: 'local', label: 'Local' }
	];

	let snapshot = $state<ClusterSnapshot | null>(null);
	let active = $state('chat');
	let error = $state<string | null>(null);

	async function poll() {
		try {
			const res = await fetch('/api/cluster');
			if (res.ok) {
				snapshot = (await res.json()) as ClusterSnapshot;
				error = null;
			} else {
				error = `cluster: HTTP ${res.status}`;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	onMount(() => {
		poll();
		const id = setInterval(poll, POLL);
		return () => clearInterval(id);
	});

	function toggleNode(name: string, action: 'stop' | 'start') {
		fetch(`/api/nodes/${name}/${action}`, { method: 'POST' }).catch(() => {});
	}

	async function send(prompt: string): Promise<GenerateResult | null> {
		const res = await fetch('/api/generate', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ prompt })
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		return (await res.json()) as GenerateResult;
	}
</script>

<svelte:head>
	<title>mixup</title>
</svelte:head>

<div class="app">
	<header>
		<span class="brand">mixup</span>
		<span class="muted">distributed MoE inference</span>
		<span class="spacer"></span>
		{#if snapshot?.leader}
			<span class="muted">leader: </span><span class="green">{snapshot.leader}</span>
		{:else}
			<span class="amber">no leader</span>
		{/if}
	</header>

	{#if error}
		<div class="banner red">{error}</div>
	{/if}

	<div class="body">
		<Sidebar nodes={snapshot?.nodes ?? []} leader={snapshot?.leader ?? null} onToggle={toggleNode} />

		<main>
			<nav>
				{#each TABS as t (t.id)}
					<button class:active={active === t.id} onclick={() => (active = t.id)}>{t.label}</button>
				{/each}
			</nav>

			<div class="content scroll">
				{#if active === 'chat'}
					<ChatTab onSend={send} />
				{:else if active === 'nodes'}
					<NodesTab nodes={snapshot?.nodes ?? []} leader={snapshot?.leader ?? null} />
				{:else if active === 'resources'}
					<ResourcesTab {snapshot} />
				{:else if active === 'experts'}
					<ExpertsTab experts={snapshot?.experts ?? null} />
				{:else if active === 'predictor'}
					<PredictorTab predictor={snapshot?.predictor ?? null} />
				{:else if active === 'summary'}
					<SummaryTab {snapshot} />
				{:else if active === 'local'}
					<LocalTab />
				{/if}
			</div>
		</main>
	</div>
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
	}

	header {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 8px 14px;
		border-bottom: 1px solid var(--border);
		background: var(--panel);
	}

	header .brand {
		font-weight: 700;
		color: var(--accent);
	}

	.spacer {
		flex: 1;
	}

	.banner {
		padding: 6px 14px;
		border-bottom: 1px solid var(--border);
		background: var(--panel-2);
	}

	.body {
		display: flex;
		flex: 1;
		min-height: 0;
	}

	main {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	nav {
		display: flex;
		gap: 2px;
		padding: 8px 14px 0;
		border-bottom: 1px solid var(--border);
	}

	nav button {
		border-radius: 6px 6px 0 0;
		border-bottom: none;
	}

	nav button.active {
		background: var(--panel);
		color: var(--accent);
		border-color: var(--border);
	}

	.content {
		flex: 1;
		padding: 14px;
	}
</style>

<script lang="ts">
	import type { LocalBenchResult } from '$lib/types';

	let result = $state<LocalBenchResult | null>(null);
	let busy = $state(false);

	async function run() {
		busy = true;
		result = null;
		try {
			const res = await fetch('/api/local', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					prompt: 'Explain mixture of experts models.',
					max_new_tokens: 60
				})
			});
			result = (await res.json()) as LocalBenchResult;
		} catch (e) {
			result = { ok: false, error: e instanceof Error ? e.message : String(e) };
		} finally {
			busy = false;
		}
	}
</script>

<div class="card">
	<p>
		Runs the whole model single-process on a GPU node (inside its container) — a
		"no distribution" baseline to compare against the distributed cluster.
	</p>
	<button class="primary" onclick={run} disabled={busy}>
		{busy ? 'running…' : 'Run local benchmark (no distribution)'}
	</button>
</div>

{#if result}
	<div class="card result">
		{#if result.ok}
			<p><b>local (no distribution)</b></p>
			<p>
				{result.tokens ?? 0} tokens in {(result.elapsed_seconds ?? 0).toFixed(2)}s =
				<span class="green">
					{((result.tokens ?? 0) / Math.max(result.elapsed_seconds ?? 0, 1e-9)).toFixed(2)} tok/s
				</span>
			</p>
			<p class="muted">{result.text}</p>
		{:else}
			<p class="red">error: {result.error}</p>
		{/if}
	</div>
{/if}

<style>
	.card p {
		margin: 6px 0;
	}

	.result {
		margin-top: 14px;
	}

	.result .muted {
		white-space: pre-wrap;
	}
</style>

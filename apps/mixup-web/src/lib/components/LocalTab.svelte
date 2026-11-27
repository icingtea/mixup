<script lang="ts">
	import type { GenerateResult, LocalBenchResult } from '$lib/types';
	import ChatTab from './ChatTab.svelte';

	async function localSend(prompt: string): Promise<GenerateResult | null> {
		const res = await fetch('/api/local', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ prompt })
		});
		const d = (await res.json()) as LocalBenchResult;
		if (!d.ok) {
			return {
				text: `error: ${d.error ?? 'unknown error'}`,
				tokens: 0,
				elapsed_seconds: 0,
				dense_seconds: null,
				dispatch_seconds: null
			};
		}
		return {
			text: d.text ?? '',
			tokens: d.tokens ?? 0,
			elapsed_seconds: d.elapsed_seconds ?? 0,
			dense_seconds: null,
			dispatch_seconds: null
		};
	}
</script>

<div class="local">
	<div class="note muted">
		single-process model · no distribution — first message loads the model, then it stays warm
	</div>
	<div class="chat-wrap">
		<ChatTab onSend={localSend} />
	</div>
</div>

<style>
	.local {
		height: 100%;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	.note {
		font-size: 12px;
	}

	.chat-wrap {
		flex: 1;
		min-height: 0;
	}
</style>

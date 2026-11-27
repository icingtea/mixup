<script lang="ts">
	import type { GenerateResult } from '$lib/types';

	interface Props {
		onSend: (prompt: string) => Promise<GenerateResult | null>;
	}

	let { onSend }: Props = $props();

	let messages = $state<{ role: 'user' | 'assistant'; text: string; meta?: string }[]>([]);
	let input = $state('');
	let busy = $state(false);

	async function submit() {
		const prompt = input.trim();
		if (!prompt || busy) return;
		input = '';
		messages = [...messages, { role: 'user', text: prompt }];
		busy = true;
		try {
			const r = await onSend(prompt);
			if (r) {
				const tokS = r.tokens / Math.max(r.elapsed_seconds, 1e-9);
				messages = [
					...messages,
					{
						role: 'assistant',
						text: r.text,
						meta: `${r.tokens} tokens in ${r.elapsed_seconds.toFixed(2)}s (${tokS.toFixed(2)} tok/s)`
					}
				];
			} else {
				messages = [...messages, { role: 'assistant', text: '(no leader reachable)' }];
			}
		} catch (e) {
			messages = [
				...messages,
				{ role: 'assistant', text: `error: ${e instanceof Error ? e.message : String(e)}` }
			];
		} finally {
			busy = false;
		}
	}
</script>

<div class="chat">
	<div class="messages scroll">
		{#each messages as m, i (i)}
			<div class="msg {m.role}">
				<span class="role">{m.role === 'user' ? 'you' : 'assistant'}</span>
				<p>{m.text}</p>
				{#if m.meta}<span class="meta">{m.meta}</span>{/if}
			</div>
		{/each}
		{#if busy}
			<div class="msg assistant">
				<span class="role">assistant</span>
				<p class="muted">…generating</p>
			</div>
		{/if}
	</div>

	<form
		class="input-row"
		onsubmit={(e) => {
			e.preventDefault();
			submit();
		}}
	>
		<input bind:value={input} placeholder="Send a prompt…" disabled={busy} />
		<button type="submit" class="primary" disabled={busy}>send</button>
	</form>
</div>

<style>
	.chat {
		display: flex;
		flex-direction: column;
		height: 100%;
		gap: 12px;
	}

	.messages {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 12px;
		min-height: 0;
	}

	.msg {
		max-width: 80%;
	}

	.msg.user {
		align-self: flex-end;
	}

	.msg.assistant {
		align-self: flex-start;
	}

	.role {
		color: var(--accent);
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.msg.user .role {
		color: var(--green);
	}

	.msg p {
		margin: 4px 0;
		background: var(--panel);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 8px 12px;
		white-space: pre-wrap;
	}

	.meta {
		color: var(--muted);
		font-size: 11px;
	}

	.input-row {
		display: flex;
		gap: 8px;
	}
</style>

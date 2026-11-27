<script lang="ts">
	import type { GenerateResult } from '$lib/types';

	interface ExpertUsage {
		top: { e: number; c: number }[];
		distinct: number;
		total: number;
		max: number;
	}

	interface Msg {
		role: 'user' | 'assistant';
		text: string;
		meta?: string;
		usage?: ExpertUsage;
	}

	interface Props {
		onSend: (prompt: string) => Promise<GenerateResult | null>;
	}

	let { onSend }: Props = $props();

	let messages = $state<Msg[]>([]);
	let input = $state('');
	let busy = $state(false);

	function computeUsage(counts: number[] | undefined): ExpertUsage | undefined {
		if (!counts || counts.length === 0) return undefined;
		const total = counts.reduce((a, c) => a + c, 0);
		if (total === 0) return undefined;
		const distinct = counts.filter((c) => c > 0).length;
		const max = Math.max(1, ...counts);
		const top = counts.map((c, e) => ({ e, c })).sort((a, b) => b.c - a.c);
		return { top, distinct, total, max };
	}

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
						meta: `${r.tokens} tokens in ${r.elapsed_seconds.toFixed(2)}s (${tokS.toFixed(2)} tok/s)`,
						usage: computeUsage(r.expert_counts)
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
				{#if m.usage}
					<details class="expert-usage">
						<summary>{m.usage.distinct} experts · {m.usage.total} activations</summary>
						<div class="expert-list">
							{#each m.usage.top as t (t.e)}
								<div class="expert-row">
									<span class="e-name">expert {t.e}</span>
									<span class="e-bar">
										<span class="e-fill" style="width:{(t.c / m.usage.max) * 100}%"></span>
									</span>
									<span class="e-count">{t.c}</span>
								</div>
							{/each}
						</div>
					</details>
				{/if}
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

	.expert-usage {
		margin-top: 4px;
		font-size: 12px;
	}

	.expert-usage summary {
		cursor: pointer;
		color: var(--muted);
		user-select: none;
	}

	.expert-usage summary:hover {
		color: var(--accent);
	}

	.expert-list {
		margin-top: 6px;
		background: var(--panel);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 6px 10px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.expert-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-variant-numeric: tabular-nums;
	}

	.e-name {
		width: 64px;
		color: var(--muted);
	}

	.e-bar {
		flex: 1;
		height: 6px;
		background: var(--panel-2);
		border-radius: 3px;
		overflow: hidden;
	}

	.e-fill {
		display: block;
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
	}

	.e-count {
		width: 28px;
		text-align: right;
		color: var(--text);
	}
</style>

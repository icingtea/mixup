<script lang="ts">
	import type { PredictorInfo } from '$lib/types';

	interface Props {
		predictor: PredictorInfo | null;
	}

	let { predictor }: Props = $props();
</script>

{#if !predictor || predictor.kind === 'none' || predictor.samples === 0}
	<p class="muted">no predictor trained yet — generate a few tokens on the leader.</p>
{:else}
	<div class="card">
		<h3>
			{predictor.kind === 'lr'
				? 'logistic regression'
				: predictor.kind === 'mlp'
					? '2-layer MLP'
					: predictor.kind}{' '}
			predictor
		</h3>
		<p class="muted">
			predicts next-layer expert top-k from <code>[h_t, one_hot(e_l), 1]</code>
		</p>

		<div class="kv"><span class="k">version</span><span>{predictor.version}</span></div>
		<div class="kv">
			<span class="k">samples</span><span>{predictor.samples}</span>
		</div>
		<div class="kv">
			<span class="k">top-1</span>
			<span>
				{predictor.top1_hits}/{predictor.samples} =
				<span class="green">{((predictor.top1_hits / predictor.samples) * 100).toFixed(1)}%</span>
			</span>
		</div>
		<div class="kv">
			<span class="k">top-3</span>
			<span>
				{predictor.top3_hits}/{predictor.samples} =
				<span class="green">{((predictor.top3_hits / predictor.samples) * 100).toFixed(1)}%</span>
			</span>
		</div>

		<div class="kv"><span class="k">feature dim</span><span>{predictor.feature_dim ?? '—'}</span></div>
		<div class="kv"><span class="k">experts</span><span>{predictor.num_experts ?? '—'}</span></div>
		{#if predictor.kind === 'mlp'}
			<div class="kv"><span class="k">hidden dim</span><span>{predictor.hidden_dim ?? '—'}</span></div>
		{/if}
		<div class="kv"><span class="k">learning rate</span><span>{predictor.learning_rate ?? '—'}</span></div>
	</div>
{/if}

<style>
	h3 {
		margin: 0;
		font-size: 14px;
		font-weight: 600;
	}

	p {
		margin: 4px 0 12px;
	}

	code {
		background: var(--panel-2);
		border-radius: 4px;
		padding: 1px 5px;
		font-size: 12px;
	}
</style>

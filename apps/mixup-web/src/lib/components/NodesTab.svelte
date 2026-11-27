<script lang="ts">
	import type { NodeInfo } from '$lib/types';
	import { fmtBytes } from '$lib/format';

	interface Props {
		nodes: NodeInfo[];
		leader: string | null;
	}

	let { nodes, leader }: Props = $props();
</script>

<div class="card">
	<table>
		<thead>
			<tr>
				<th>node</th>
				<th>role</th>
				<th>leader</th>
				<th>exec</th>
				<th>experts</th>
				<th>cpu%</th>
				<th>gpu free</th>
				<th>rtt</th>
				<th>tokens</th>
			</tr>
		</thead>
		<tbody>
			{#each nodes as n (n.name)}
				<tr>
					<td>
						<span class="dot {n.reachable ? 'ok' : 'down'}"></span>{n.name}
					</td>
					<td>{n.role}</td>
					<td>{n.name === leader ? '★' : ''}</td>
					<td>{n.execution ?? '?'}</td>
					<td>{n.ownedExperts.length}</td>
					<td>{((n.resources?.cpu_util ?? 0) * 100).toFixed(1)}</td>
					<td>{fmtBytes(n.resources?.gpu_free_bytes)}</td>
					<td>{n.rttMs != null ? `${n.rttMs.toFixed(0)}ms` : '—'}</td>
					<td>{n.totalTokens}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

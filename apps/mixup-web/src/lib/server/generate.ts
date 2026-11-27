// Failover-aware generation, mirroring the REPL's ClusterClient: discover the leader
// (highest epoch), wait for it to be ready, POST /generate with no timeout, and retry
// against the newly-discovered leader if the connection drops mid-request.

import type { GenerateResult } from '$lib/types';
import { discoverLeader } from './cluster';
import { loadStartup, nodeUrls } from './config';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitReady(url: string, timeoutMs = 30000): Promise<boolean> {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		try {
			const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
			const j = (await res.json()) as { ready?: boolean };
			if (j.ready) return true;
		} catch {
			// not ready / unreachable
		}
		await sleep(1000);
	}
	return false;
}

export async function generate(prompt: string, maxNewTokens: number): Promise<GenerateResult> {
	const urls = nodeUrls(loadStartup());
	for (let attempt = 0; attempt < 10; attempt++) {
		const leader = await discoverLeader();
		if (!leader) {
			await sleep(2000);
			continue;
		}
		const url = urls[leader];
		if (!(await waitReady(url))) {
			await sleep(2000);
			continue;
		}
		try {
			const res = await fetch(`${url}/generate`, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens })
				// no timeout: generation is long; a killed leader throws on the dropped
				// connection, which we catch and retry against the new leader.
			});
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			return (await res.json()) as GenerateResult;
		} catch {
			await sleep(2000); // wait for failover, then re-discover
		}
	}
	throw new Error('could not reach a leader after retries');
}

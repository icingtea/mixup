// Node-discovery config: mirrors `mixup_core.hosts.node_urls` and
// `mixup_core.startup_config.load_startup_config` on the host side.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const EXTERNAL_PORT_BASE = 9000;

export interface StartupConfig {
	initial_leader: string;
	nodes: string[];
	max_new_tokens?: number;
}

const DEFAULT_MAX_NEW_TOKENS = 500;

function findConfig(): string {
	const env = process.env.MIXUP_CONFIG;
	if (env) {
		const p = resolve(env);
		if (existsSync(p)) return p;
		throw new Error(`MIXUP_CONFIG not found: ${env}`);
	}
	let dir = process.cwd();
	for (let i = 0; i < 8; i++) {
		for (const name of ['lab_config.json', 'startup_config.json']) {
			const p = join(dir, name);
			if (existsSync(p)) return p;
		}
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	throw new Error('no lab_config.json / startup_config.json found (set MIXUP_CONFIG)');
}

export function loadStartup(): StartupConfig {
	const raw = JSON.parse(readFileSync(findConfig(), 'utf8'));
	return (raw.startup ?? raw) as StartupConfig;
}

/** Default generation length from the config (``max_new_tokens``), else 500. */
export function maxNewTokens(): number {
	return loadStartup().max_new_tokens ?? DEFAULT_MAX_NEW_TOKENS;
}

/** The first node with a GPU execution profile (used for the local baseline). */
export function gpuNode(): string {
	const raw = JSON.parse(readFileSync(findConfig(), 'utf8'));
	const profiles = (raw.profiles ?? {}) as Record<string, { execution?: string }>;
	for (const [name, p] of Object.entries(profiles)) {
		if (p?.execution === 'gpu') return name;
	}
	return loadStartup().initial_leader;
}

/** Node name -> host HTTP URL (leader on :9000, others on :9001, ...). */
export function nodeUrls(cfg: StartupConfig): Record<string, string> {
	const urls: Record<string, string> = {};
	urls[cfg.initial_leader] = `http://localhost:${EXTERNAL_PORT_BASE}`;
	const others = cfg.nodes.filter((n) => n !== cfg.initial_leader);
	others.forEach((n, i) => {
		urls[n] = `http://localhost:${EXTERNAL_PORT_BASE + i + 1}`;
	});
	return urls;
}

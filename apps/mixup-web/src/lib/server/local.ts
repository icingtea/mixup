// The "no distribution" chat backend: a single-process full model running on a GPU node,
// kept loaded across messages so each turn is fast (no per-message model reload).

import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface, type Interface } from 'node:readline';

import type { LocalBenchResult } from '$lib/types';
import { gpuNode } from './config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SCRIPT = resolve(__dirname, '../../../scripts/local_chat.py');
const LOAD_TIMEOUT_MS = 300000;

let proc: ChildProcess | null = null;
let rl: Interface | null = null;
let pending: ((r: LocalBenchResult) => void)[] = [];
let chain: Promise<LocalBenchResult> = Promise.resolve({ ok: true, text: '', tokens: 0, elapsed_seconds: 0 });

function failPending(reason: string) {
	for (const resolve of pending.splice(0)) {
		resolve({ ok: false, error: reason });
	}
}

function ensureSpawned() {
	if (proc && rl && proc.stdin?.writable && proc.exitCode === null) return;
	const node = gpuNode();
	const code = readFileSync(SCRIPT, 'utf8');
	const p = spawn(
		'docker',
		['exec', '-i', node, '/usr/local/app/.venv/bin/python', '-c', code],
		{ stdio: ['pipe', 'pipe', 'pipe'] }
	);
	proc = p;
	const r = createInterface({ input: p.stdout });
	rl = r;

	r.on('line', (line) => {
		const resolveReq = pending.shift();
		if (!resolveReq) return;
		try {
			resolveReq({ ...(JSON.parse(line) as LocalBenchResult), ok: true });
		} catch {
			resolveReq({ ok: false, error: `unexpected output: ${line.slice(0, 300)}` });
		}
	});

	let errTail = '';
	p.stderr.on('data', (d) => {
		errTail = (errTail + d.toString()).slice(-500);
	});
	p.on('exit', () => {
		if (proc === p) {
			proc = null;
			rl = null;
		}
		failPending(`local model exited${errTail ? `: ${errTail.slice(-200)}` : ''}`);
	});
	p.on('error', (e) => {
		if (proc === p) {
			proc = null;
			rl = null;
		}
		failPending(e.message);
	});
}

export function localChat(prompt: string, maxNewTokens = 60): Promise<LocalBenchResult> {
	const task = () =>
		new Promise<LocalBenchResult>((resolve) => {
			ensureSpawned();
			const p = proc;
			if (!p || !p.stdin?.writable || p.exitCode !== null) {
				resolve({ ok: false, error: 'local model unavailable' });
				return;
			}
			let timer: NodeJS.Timeout;
			const wrapped = (r: LocalBenchResult) => {
				clearTimeout(timer);
				resolve(r);
			};
			timer = setTimeout(() => {
				const i = pending.indexOf(wrapped);
				if (i >= 0) {
					pending.splice(i, 1);
					wrapped({ ok: false, error: 'timed out waiting for local model' });
				}
			}, LOAD_TIMEOUT_MS);
			pending.push(wrapped);
			p.stdin.write(`${JSON.stringify({ prompt, max_new_tokens: maxNewTokens })}\n`);
		});
	chain = chain.then(task, task);
	return chain;
}

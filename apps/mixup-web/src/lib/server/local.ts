// The "no distribution" baseline: run the whole model single-process on a GPU node,
// inside its container (which has torch + the model), and return the result for a
// side-by-side tok/s comparison with the distributed path.

import { execFile } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import type { LocalBenchResult } from '$lib/types';
import { gpuNode } from './config';

const execFileP = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const SCRIPT = resolve(__dirname, '../../../scripts/local_bench.py');

export async function localBench(prompt: string, maxNewTokens = 60): Promise<LocalBenchResult> {
	const node = gpuNode();
	const code = readFileSync(SCRIPT, 'utf8');
	const payload = JSON.stringify({ prompt, max_new_tokens: maxNewTokens });
	try {
		const { stdout, stderr } = await execFileP(
			'docker',
			['exec', node, '/usr/local/app/.venv/bin/python', '-c', code, payload],
			{ timeout: 240000, maxBuffer: 16 * 1024 * 1024 }
		);
		const line = (stdout.trim().split('\n').pop() ?? '').trim();
		try {
			const parsed = JSON.parse(line) as LocalBenchResult;
			return { ...parsed, ok: true };
		} catch {
			return {
				ok: false,
				error: `unexpected output${stderr ? ` (stderr: ${stderr.slice(0, 200)})` : ''}: ${line.slice(0, 300)}`
			};
		}
	} catch (err) {
		return { ok: false, error: err instanceof Error ? err.message : String(err) };
	}
}

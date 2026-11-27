// Node lifecycle control: `docker stop` / `docker start`, like the TUI's kill/start
// toggles. Callers must validate the container name against the known node set. Remote
// nodes are driven through `docker -H <daemon>` (from the config's per-node docker_host).

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import { dockerHostFor } from './config';

const execFileP = promisify(execFile);

export async function docker(action: 'stop' | 'start', name: string): Promise<string> {
	const host = dockerHostFor(name);
	const args = host ? ['-H', host, action, name] : [action, name];
	const { stdout } = await execFileP('docker', args, { timeout: 20000 });
	return stdout.trim();
}

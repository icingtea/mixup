// Node lifecycle control: `docker stop` / `docker start`, like the TUI's kill/start
// toggles. Callers must validate the container name against the known node set.

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileP = promisify(execFile);

export async function docker(action: 'stop' | 'start', name: string): Promise<string> {
	const { stdout } = await execFileP('docker', [action, name], { timeout: 20000 });
	return stdout.trim();
}

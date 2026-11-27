import { error, json } from '@sveltejs/kit';

import { nodeNames } from '$lib/server/cluster';
import { docker } from '$lib/server/docker';

export async function POST({ params }) {
	const { name, action } = params;
	if (action !== 'stop' && action !== 'start') throw error(400, 'action must be stop or start');
	if (!nodeNames().includes(name)) throw error(404, `unknown node: ${name}`);
	await docker(action, name);
	return json({ ok: true, action, name });
}

import { json } from '@sveltejs/kit';

import { localBench } from '$lib/server/local';

export async function POST({ request }) {
	const body = (await request.json()) as { prompt?: string; max_new_tokens?: number };
	const result = await localBench(
		body.prompt ?? 'Explain mixture of experts models.',
		body.max_new_tokens ?? 60
	);
	return json(result);
}

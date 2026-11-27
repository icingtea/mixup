import { json } from '@sveltejs/kit';

import { generate } from '$lib/server/generate';

export async function POST({ request }) {
	const body = (await request.json()) as { prompt?: string; max_new_tokens?: number };
	if (!body.prompt) return json({ error: 'prompt required' }, { status: 400 });
	return json(await generate(body.prompt, body.max_new_tokens ?? 80));
}

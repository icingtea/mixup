import { json } from '@sveltejs/kit';

import { localChat } from '$lib/server/local';

export async function POST({ request }) {
	const body = (await request.json()) as { prompt?: string; max_new_tokens?: number };
	const result = await localChat(body.prompt ?? '', body.max_new_tokens ?? 60);
	return json(result);
}

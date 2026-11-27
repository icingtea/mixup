import { json } from '@sveltejs/kit';

import { maxNewTokens } from '$lib/server/config';
import { localChat } from '$lib/server/local';

export async function POST({ request }) {
	const body = (await request.json()) as { prompt?: string; max_new_tokens?: number };
	const result = await localChat(body.prompt ?? '', body.max_new_tokens ?? maxNewTokens());
	return json(result);
}

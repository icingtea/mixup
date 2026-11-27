import { json } from '@sveltejs/kit';

import { snapshot } from '$lib/server/cluster';

export async function GET() {
	return json(await snapshot());
}

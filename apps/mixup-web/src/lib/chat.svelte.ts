// Shared, module-scoped chat state so history survives tab switches (the ChatTab
// component is remounted, but the store lives for the page's lifetime). The main chat and
// the local (no-distribution) chat each keep their own store.

import type { RoutingAssignment } from './types';

export interface ExpertUsage {
	top: { e: number; c: number }[];
	distinct: number;
	total: number;
	max: number;
}

export interface Msg {
	role: 'user' | 'assistant';
	text: string;
	meta?: string;
	usage?: ExpertUsage;
	routing?: Record<string, RoutingAssignment>;
}

export interface ChatState {
	messages: Msg[];
	busy: boolean;
	input: string;
}

export const chat = $state<ChatState>({ messages: [], busy: false, input: '' });
export const localChat = $state<ChatState>({ messages: [], busy: false, input: '' });

export function computeUsage(counts: number[] | undefined): ExpertUsage | undefined {
	if (!counts || counts.length === 0) return undefined;
	const total = counts.reduce((a, c) => a + c, 0);
	if (total === 0) return undefined;
	const distinct = counts.filter((c) => c > 0).length;
	const max = Math.max(1, ...counts);
	const top = counts.map((c, e) => ({ e, c })).sort((a, b) => b.c - a.c);
	return { top, distinct, total, max };
}

export function nodeOf(routing: Record<string, RoutingAssignment> | undefined, expert: number): string {
	return routing?.[expert]?.node ?? '—';
}

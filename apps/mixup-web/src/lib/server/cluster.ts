// Cluster aggregation: probes every node, discovers the leader, records per-node
// resource time series (including control-plane RTT), and snapshots the leader's
// predictor/expert state. Replicates what the Textual TUI's poller did.

import { performance } from 'node:perf_hooks';

import type {
	ClusterSnapshot,
	DeviceProfile,
	ExpertStats,
	HistorySeries,
	NodeInfo,
	PredictorInfo,
	ResidencyInfo,
	Resources,
	RoutingAssignment
} from '$lib/types';
import { loadStartup, nodeUrls } from './config';

const DISCOVER_TIMEOUT_MS = 2000;
const HISTORY_WINDOW = 300; // samples (~5 min at 1s poll)

let _cfg: ReturnType<typeof loadStartup> | null = null;
let _urls: Record<string, string> | null = null;
const _history: Record<string, HistorySeries> = {};

function urls(): Record<string, string> {
	if (!_cfg || !_urls) {
		_cfg = loadStartup();
		_urls = nodeUrls(_cfg);
		for (const name of Object.keys(_urls)) {
			_history[name] = { t: [], cpu: [], mem: [], gpu: [], rtt: [] };
		}
	}
	return _urls;
}

async function getJson(url: string, timeoutMs: number): Promise<unknown> {
	const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	return res.json();
}

async function timedGet(url: string): Promise<{ json: unknown; rttMs: number }> {
	const start = performance.now();
	const json = await getJson(url, DISCOVER_TIMEOUT_MS);
	return { json, rttMs: performance.now() - start };
}

function emptyHistory(name: string): HistorySeries {
	return _history[name] ?? { t: [], cpu: [], mem: [], gpu: [], rtt: [] };
}

function push(series: HistorySeries, cpu: number, mem: number | null, gpu: number | null, rtt: number | null) {
	series.t.push(Date.now());
	series.cpu.push(cpu);
	series.mem.push(mem ?? 0);
	series.gpu.push(gpu ?? 0);
	series.rtt.push(rtt ?? 0);
	if (series.t.length > HISTORY_WINDOW) {
		series.t = series.t.slice(-HISTORY_WINDOW);
		series.cpu = series.cpu.slice(-HISTORY_WINDOW);
		series.mem = series.mem.slice(-HISTORY_WINDOW);
		series.gpu = series.gpu.slice(-HISTORY_WINDOW);
		series.rtt = series.rtt.slice(-HISTORY_WINDOW);
	}
}

export function nodeNames(): string[] {
	return Object.keys(urls());
}

/** Discover the leader: the node id with the highest epoch across /leader projections. */
export async function discoverLeader(): Promise<string | null> {
	let best: string | null = null;
	let bestEpoch = -1;
	await Promise.all(
		Object.values(urls()).map(async (url) => {
			try {
				const data = (await getJson(`${url}/leader`, DISCOVER_TIMEOUT_MS)) as {
					leader?: string;
					epoch?: number;
				};
				if (data.leader && (data.epoch ?? 0) > bestEpoch) {
					bestEpoch = data.epoch ?? 0;
					best = data.leader;
				}
			} catch {
				// unreachable node
			}
		})
	);
	return best && best in urls() ? best : null;
}

async function fetchNodeInfo(name: string, url: string): Promise<NodeInfo> {
	const info: NodeInfo = {
		name,
		url,
		reachable: false,
		ready: false,
		role: '?',
		execution: null,
		ownedExperts: [],
		totalTokens: 0,
		routingEpoch: null,
		resources: null,
		profile: null,
		residency: null,
		rttMs: null
	};

	let res: Resources | null = null;
	let rttMs: number | null = null;
	try {
		const { json, rttMs: rtt } = await timedGet(`${url}/metrics`);
		rttMs = rtt;
		const m = json as {
			role?: string;
			execution?: string | null;
			owned_experts?: number[];
			resources?: Resources;
			profile?: DeviceProfile | null;
			residency?: ResidencyInfo | null;
			stats?: { total_tokens?: number };
			routing_epoch?: number | null;
		};
		info.reachable = true;
		info.role = m.role ?? '?';
		info.execution = m.execution ?? null;
		info.ownedExperts = m.owned_experts ?? [];
		info.totalTokens = m.stats?.total_tokens ?? 0;
		info.routingEpoch = m.routing_epoch ?? null;
		res = m.resources ?? null;
		info.resources = res;
		info.profile = m.profile ?? null;
		info.residency = m.residency ?? null;
	} catch {
		// keep unreachable
	}

	try {
		const h = (await getJson(`${url}/health`, DISCOVER_TIMEOUT_MS)) as { ready?: boolean };
		info.ready = !!h.ready;
	} catch {
		// unreachable
	}

	info.rttMs = rttMs;
	push(
		emptyHistory(name),
		res?.cpu_util ?? 0,
		res?.mem_available_bytes ?? null,
		res?.gpu_free_bytes ?? null,
		rttMs
	);
	return info;
}

export async function snapshot(): Promise<ClusterSnapshot> {
	const map = urls();
	const nodes = await Promise.all(Object.entries(map).map(([name, url]) => fetchNodeInfo(name, url)));

	const leader = await discoverLeader();

	let predictor: PredictorInfo | null = null;
	let experts: ExpertStats | null = null;
	let routing: { epoch: number | null; assignment: Record<string, RoutingAssignment> } = {
		epoch: null,
		assignment: {}
	};

	if (leader) {
		const lurl = map[leader];
		try {
			predictor = (await getJson(`${lurl}/predictor`, DISCOVER_TIMEOUT_MS)) as PredictorInfo;
		} catch {
			predictor = null;
		}
		try {
			experts = (await getJson(`${lurl}/experts`, DISCOVER_TIMEOUT_MS)) as ExpertStats;
		} catch {
			experts = null;
		}
		try {
			routing = (await getJson(`${lurl}/routing`, DISCOVER_TIMEOUT_MS)) as typeof routing;
		} catch {
			// keep empty
		}
	}

	return {
		leader,
		nodes,
		predictor,
		experts,
		routing,
		history: Object.fromEntries(Object.keys(map).map((n) => [n, emptyHistory(n)]))
	};
}

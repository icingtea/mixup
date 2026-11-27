// Shared shapes between the SvelteKit server routes and the frontend.

export interface Resources {
	cpu_util: number;
	cpu_util_stability: number;
	mem_available_bytes: number | null;
	mem_stability: number;
	gpu_total_bytes: number | null;
	gpu_free_bytes: number | null;
}

export interface NodeInfo {
	name: string;
	url: string;
	reachable: boolean;
	ready: boolean;
	role: string;
	execution: string | null;
	ownedExperts: number[];
	totalTokens: number;
	routingEpoch: number | null;
	resources: Resources | null;
	rttMs: number | null;
}

export interface PredictorInfo {
	kind: string;
	version: number;
	samples: number;
	top1_hits: number;
	top3_hits: number;
}

export interface ExpertStats {
	counts: number[];
	num_experts: number;
}

export interface RoutingAssignment {
	node: string;
	slot: number;
	backup: string | null;
}

export interface HistorySeries {
	t: number[];
	cpu: number[];
	mem: number[]; // bytes available
	gpu: number[]; // bytes free
	rtt: number[]; // ms
}

export interface ClusterSnapshot {
	leader: string | null;
	nodes: NodeInfo[];
	predictor: PredictorInfo | null;
	experts: ExpertStats | null;
	routing: { epoch: number | null; assignment: Record<string, RoutingAssignment> };
	history: Record<string, HistorySeries>;
}

export interface GenerateResult {
	text: string;
	tokens: number;
	elapsed_seconds: number;
	dense_seconds: number | null;
	dispatch_seconds: number | null;
}

export interface LocalBenchResult {
	ok: boolean;
	error?: string;
	text?: string;
	tokens?: number;
	elapsed_seconds?: number;
}

// Shared shapes between the SvelteKit server routes and the frontend.

export interface Resources {
	cpu_util: number;
	cpu_util_stability: number;
	mem_available_bytes: number | null;
	mem_stability: number;
	gpu_total_bytes: number | null;
	gpu_free_bytes: number | null;
}

export interface DeviceProfile {
	node_id: string;
	execution: string;
	cpu_cores: number;
	cpu_gflops: number;
	gpu_tflops: number | null;
	expert_fixed_ms: number;
	expert_per_ms: number;
	ram_bytes: number;
	vram_bytes: number | null;
	expert_budget_bytes: number | null;
	disk_read_mbps: number;
	disk_write_mbps: number;
	gpu_cpu_bw_mbps: number | null;
	can_lead: boolean;
	lead_score: number;
}

export interface ResidencyInfo {
	execution_tier: string;
	budget_bytes: number | null;
	used_bytes: number;
	free_bytes: number;
	free_ratio: number;
	resident_count: number;
	prefetch_threshold?: number | null;
	config: {
		gamma: number;
		delta: number;
		lambda: number;
		theta_base: number;
		theta_mem: number;
		decay: number;
	};
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
	profile: DeviceProfile | null;
	residency: ResidencyInfo | null;
	rttMs: number | null;
}

export interface PredictorInfo {
	kind: string;
	version: number;
	samples: number;
	top1_hits: number;
	top3_hits: number;
	feature_dim?: number;
	num_experts?: number;
	hidden_dim?: number;
	learning_rate?: number;
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

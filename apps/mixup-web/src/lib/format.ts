export function fmtBytes(b: number | null | undefined): string {
	if (b == null) return '—';
	if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
	if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
	return `${(b / 1e3).toFixed(0)} KB`;
}

export function fmtMs(v: number | null | undefined): string {
	if (v == null) return '—';
	return `${v.toFixed(3)} ms`;
}

export function fmtMbps(v: number | null | undefined): string {
	if (v == null) return '—';
	return `${v.toFixed(0)} MB/s`;
}

export function fmtRatio(v: number | null | undefined): string {
	if (v == null) return '—';
	return v.toFixed(3);
}

export function fmtPct(v: number | null | undefined): string {
	if (v == null) return '—';
	return `${(v * 100).toFixed(1)}%`;
}

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

export function fmtRanges(ids: number[]): string {
	/** Collapse a sorted list of ints into ranges, e.g. [0,1,2,5,6] -> "0-2, 5-6". */
	if (ids.length === 0) return '—';
	const s = [...ids].sort((a, b) => a - b);
	const ranges: string[] = [];
	let start = s[0];
	for (let i = 1; i <= s.length; i++) {
		if (i < s.length && s[i] === s[i - 1] + 1) continue;
		ranges.push(start === s[i - 1] ? `${start}` : `${start}–${s[i - 1]}`);
		if (i < s.length) start = s[i];
	}
	return ranges.join(', ');
}

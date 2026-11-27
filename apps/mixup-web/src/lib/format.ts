export function fmtBytes(b: number | null | undefined): string {
	if (b == null) return '—';
	if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
	if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
	return `${(b / 1e3).toFixed(0)} KB`;
}

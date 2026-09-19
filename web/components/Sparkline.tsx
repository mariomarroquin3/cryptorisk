/** Tiny inline trend line. Green when the series ends above where it started,
 * red otherwise. Pure SVG: no chart library, cheap to render in a table row. */
export function Sparkline({
  values,
  width = 84,
  height = 22,
}: {
  values: number[];
  width?: number;
  height?: number;
}) {
  const v = values.filter((x) => Number.isFinite(x));
  if (v.length < 2) return <span className="text-muted">n/a</span>;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  const pts = v
    .map((x, i) => `${((i / (v.length - 1)) * (width - 2) + 1).toFixed(1)},${(height - 2 - ((x - lo) / span) * (height - 4)).toFixed(1)}`)
    .join(" ");
  const up = v[v.length - 1] >= v[0];
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden className="inline-block align-middle">
      <polyline
        points={pts}
        fill="none"
        stroke={up ? "var(--green)" : "var(--red)"}
        strokeWidth={1.4}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

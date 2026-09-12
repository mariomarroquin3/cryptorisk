export function MetricCard({
  label,
  value,
  delta,
  deltaColor = "text-muted",
  title,
}: {
  label: string;
  value: string;
  delta?: string;
  deltaColor?: string;
  title?: string;
}) {
  return (
    <div className="rounded border border-grid bg-panel px-4 py-3" title={title}>
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-text">{value}</div>
      {delta && <div className={`mt-1 text-xs ${deltaColor}`}>{delta}</div>}
    </div>
  );
}

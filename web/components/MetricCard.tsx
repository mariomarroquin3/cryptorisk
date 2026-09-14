export function MetricCard({
  label,
  value,
  delta,
  deltaColor = "text-muted",
  accent,
  title,
}: {
  label: string;
  value: string;
  delta?: string;
  deltaColor?: string;
  /** Optional top accent color (a CSS color, e.g. "var(--green)") for the
   * card's most important metric on a page (e.g. the live price). */
  accent?: string;
  title?: string;
}) {
  return (
    <div className="card card-interactive relative overflow-hidden px-4 py-3" title={title}>
      {accent && (
        <div className="absolute inset-x-0 top-0 h-0.5" style={{ background: accent }} />
      )}
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1.5 font-display text-2xl font-semibold text-text">{value}</div>
      {delta && <div className={`mt-1 text-xs ${deltaColor}`}>{delta}</div>}
    </div>
  );
}

const ZONE_STYLES: Record<string, string> = {
  green: "bg-green/10 text-green border-green/40",
  amber: "bg-amber/10 text-amber border-amber/40",
  red: "bg-red/10 text-red border-red/40",
};

const ZONE_LABELS: Record<string, string> = {
  green: "Green",
  amber: "Amber",
  red: "Red",
};

/** Basel traffic-light zone (green/amber/red), computed server-side from the
 * 250-day exception count -- see `backtest/coverage.py::basel_zone_and_addon`,
 * the single source of truth. Every endpoint that has an exception count
 * (coverage, portfolio eval, capital) exposes `basel_zone` directly so this
 * component never has to re-derive it from thresholds. */
export function ZoneBadge({ zone }: { zone: string | null | undefined }) {
  if (!zone || !(zone in ZONE_STYLES)) {
    return <span className="text-muted">n/a</span>;
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide ${ZONE_STYLES[zone]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${zone === "green" ? "bg-green" : zone === "amber" ? "bg-amber" : "bg-red"}`} />
      {ZONE_LABELS[zone]}
    </span>
  );
}

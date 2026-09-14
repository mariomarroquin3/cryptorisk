import { ZoneBadge } from "@/components/ZoneBadge";

/** Same shell as MetricCard, but the headline value is the Basel
 * traffic-light badge instead of a formatted number/string. */
export function RiskZoneCard({
  label,
  zone,
  caption,
}: {
  label: string;
  zone: string | null | undefined;
  caption?: string;
}) {
  return (
    <div className="card card-interactive relative overflow-hidden px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-2">
        <ZoneBadge zone={zone} />
      </div>
      {caption && <div className="mt-1.5 text-xs text-muted">{caption}</div>}
    </div>
  );
}

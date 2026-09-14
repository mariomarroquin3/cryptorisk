/** Shimmering placeholder bar for content that's still loading over SWR. */
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded-md ${className}`} />;
}

/** Placeholder for a chart/plot area of a given pixel height. */
export function ChartSkeleton({ height = 300 }: { height?: number }) {
  return (
    <div className="card p-4">
      <Skeleton className="h-4 w-48" />
      <div className="mt-4 skeleton rounded-lg" style={{ height }} />
    </div>
  );
}

/** Placeholder shaped like DataTable: a header row + N body rows. */
export function TableSkeleton({ cols = 5, rows = 4 }: { cols?: number; rows?: number }) {
  return (
    <div className="overflow-hidden rounded-lg border border-grid">
      <div className="flex gap-3 border-b border-grid bg-panel-2 px-3 py-2.5">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-3 flex-1" />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-3 border-b border-grid/60 px-3 py-2.5 last:border-0">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={c} className="h-3 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

/** Placeholder shaped like a MetricCard grid cell. */
export function MetricCardSkeleton() {
  return (
    <div className="card relative overflow-hidden px-4 py-3">
      <Skeleton className="h-3 w-20" />
      <Skeleton className="mt-2.5 h-6 w-28" />
    </div>
  );
}

import { HelpTip } from "@/components/HelpTip";
import { TableSkeleton } from "@/components/Skeleton";

export interface Column<T> {
  key: keyof T;
  label: string;
  align?: "left" | "right";
  render?: (row: T) => React.ReactNode;
  /** 0..1 p-value-ish column: green when high, red when low. */
  pShade?: boolean;
  /** Glossary text shown in a hover/tap tooltip next to the header, for
   * jargon (p-values, test names, abbreviations) a non-quant reader can't
   * decode from the column label alone. */
  help?: string;
}

export function DataTable<T>({
  columns,
  rows,
  keyField,
  loading = false,
}: {
  columns: Column<T>[];
  rows: T[];
  keyField: keyof T | ((row: T) => string);
  /** True while the underlying SWR fetch hasn't resolved yet (data is
   * `undefined`) -- renders a shimmering skeleton instead of "No data",
   * which would otherwise flash misleadingly during the initial load. */
  loading?: boolean;
}) {
  if (loading) {
    return <TableSkeleton cols={columns.length} rows={4} />;
  }
  if (rows.length === 0) {
    return <div className="text-sm text-muted">No data for this selection.</div>;
  }
  const rowKey = typeof keyField === "function" ? keyField : (row: T) => String(row[keyField]);
  return (
    <div className="scrollbar-thin overflow-x-auto rounded-lg border border-grid">
      <table className="w-full min-w-max text-left text-sm">
        <thead>
          <tr className="border-b border-grid bg-panel-2 text-muted">
            {columns.map((c, ci) => (
              <th
                key={ci}
                className={`px-3 py-2.5 text-xs font-semibold tracking-wide uppercase ${c.align === "right" ? "text-right" : "text-left"}`}
              >
                {c.label}
                {c.help && <HelpTip text={c.help} />}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={`${rowKey(row)}-${i}`}
              className="border-b border-grid/60 transition-colors last:border-0 hover:bg-panel-2/50"
            >
              {columns.map((c, ci) => {
                const raw = row[c.key];
                let bg: string | undefined;
                if (c.pShade && typeof raw === "number") {
                  const t = Math.max(0, Math.min(1, raw / 0.2));
                  bg = `color-mix(in srgb, var(--red) ${(1 - t) * 35}%, var(--green) ${t * 35}%, transparent)`;
                }
                return (
                  <td
                    key={ci}
                    className={`px-3 py-1.5 ${c.align === "right" ? "text-right" : "text-left"}`}
                    style={bg ? { backgroundColor: bg } : undefined}
                  >
                    {c.render ? c.render(row) : formatCell(raw)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(v: unknown): React.ReactNode {
  if (v == null) return <span className="text-muted">n/a</span>;
  if (typeof v === "boolean") return v ? "✓" : "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(4);
  return String(v);
}

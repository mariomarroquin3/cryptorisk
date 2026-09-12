export interface Column<T> {
  key: keyof T;
  label: string;
  align?: "left" | "right";
  render?: (row: T) => React.ReactNode;
  /** 0..1 p-value-ish column: green when high, red when low. */
  pShade?: boolean;
}

export function DataTable<T>({
  columns,
  rows,
  keyField,
}: {
  columns: Column<T>[];
  rows: T[];
  keyField: keyof T | ((row: T) => string);
}) {
  if (rows.length === 0) {
    return <div className="text-sm text-muted">No data for this selection.</div>;
  }
  const rowKey = typeof keyField === "function" ? keyField : (row: T) => String(row[keyField]);
  return (
    <div className="scrollbar-thin overflow-x-auto rounded border border-grid">
      <table className="w-full min-w-max text-left text-sm">
        <thead>
          <tr className="border-b border-grid bg-panel text-muted">
            {columns.map((c) => (
              <th
                key={String(c.key)}
                className={`px-3 py-2 font-medium ${c.align === "right" ? "text-right" : "text-left"}`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`${rowKey(row)}-${i}`} className="border-b border-grid/60 last:border-0">
              {columns.map((c) => {
                const raw = row[c.key];
                let bg: string | undefined;
                if (c.pShade && typeof raw === "number") {
                  const t = Math.max(0, Math.min(1, raw / 0.2));
                  bg = `color-mix(in srgb, var(--red) ${(1 - t) * 35}%, var(--green) ${t * 35}%, transparent)`;
                }
                return (
                  <td
                    key={String(c.key)}
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

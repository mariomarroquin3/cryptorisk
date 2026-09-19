"use client";

import { useMemo, useState } from "react";
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

/** A toggle chip that narrows the table to rows passing `test`. */
export interface TableFilter<T> {
  label: string;
  test: (row: T) => boolean;
}

type SortState = { col: number; dir: "asc" | "desc" } | null;

function compare(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1; // missing values always sort last
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean") return Number(a) - Number(b);
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function DataTable<T>({
  columns,
  rows,
  keyField,
  loading = false,
  filterKey,
  filters = [],
}: {
  columns: Column<T>[];
  rows: T[];
  keyField: keyof T | ((row: T) => string);
  /** True while the underlying SWR fetch hasn't resolved yet (data is
   * `undefined`) -- renders a shimmering skeleton instead of "No data",
   * which would otherwise flash misleadingly during the initial load. */
  loading?: boolean;
  /** When set, shows a search box matching this field's text. */
  filterKey?: keyof T;
  /** Toggle chips; several active chips combine with AND. */
  filters?: TableFilter<T>[];
}) {
  const [sort, setSort] = useState<SortState>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState<Set<number>>(new Set());

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    let out = rows.filter((row) => {
      if (filterKey && q && !String(row[filterKey] ?? "").toLowerCase().includes(q)) return false;
      for (const i of active) if (!filters[i].test(row)) return false;
      return true;
    });
    if (sort) {
      const key = columns[sort.col].key;
      const sign = sort.dir === "asc" ? 1 : -1;
      out = [...out].sort((x, y) => {
        const a = x[key];
        const b = y[key];
        // keep missing values last regardless of direction
        if (a == null || b == null) return compare(a, b);
        return sign * compare(a, b);
      });
    }
    return out;
  }, [rows, columns, sort, query, active, filterKey, filters]);

  if (loading) {
    return <TableSkeleton cols={columns.length} rows={4} />;
  }
  if (rows.length === 0) {
    return <div className="text-sm text-muted">No data for this selection.</div>;
  }
  const rowKey = typeof keyField === "function" ? keyField : (row: T) => String(row[keyField]);
  const toggle = (i: number) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  const clickSort = (ci: number) =>
    setSort((s) => (s?.col !== ci ? { col: ci, dir: "asc" } : s.dir === "asc" ? { col: ci, dir: "desc" } : null));
  const hasControls = filterKey != null || filters.length > 0;

  return (
    <div className="space-y-2">
      {hasControls && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {filterKey && (
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Filter ${String(filterKey)}…`}
              aria-label={`Filter by ${String(filterKey)}`}
              className="w-40 rounded-md border border-grid bg-panel px-2.5 py-1 text-text placeholder:text-muted/60 focus:border-amber focus:outline-none"
            />
          )}
          {filters.map((f, i) => (
            <button
              key={f.label}
              type="button"
              aria-pressed={active.has(i)}
              onClick={() => toggle(i)}
              className={`rounded-full border px-2.5 py-1 transition-colors ${
                active.has(i)
                  ? "border-amber bg-amber/15 text-amber"
                  : "border-grid text-muted hover:border-muted hover:text-text"
              }`}
            >
              {f.label}
            </button>
          ))}
          {(query || active.size > 0) && (
            <span className="text-muted">
              {shown.length} of {rows.length}
            </span>
          )}
        </div>
      )}
      <div className="scrollbar-thin overflow-x-auto rounded-lg border border-grid">
        <table className="w-full min-w-max text-left text-sm">
          <thead>
            <tr className="border-b border-grid bg-panel-2 text-muted">
              {columns.map((c, ci) => {
                const dir = sort?.col === ci ? sort.dir : null;
                return (
                  <th
                    key={ci}
                    aria-sort={dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}
                    className={`px-3 py-2.5 text-xs font-semibold tracking-wide uppercase ${c.align === "right" ? "text-right" : "text-left"}`}
                  >
                    <button
                      type="button"
                      onClick={() => clickSort(ci)}
                      className={`inline-flex items-center gap-1 uppercase tracking-wide hover:text-text ${dir ? "text-text" : ""}`}
                      title="Click to sort"
                    >
                      {c.label}
                      <span aria-hidden className="text-[0.6rem]">
                        {dir === "asc" ? "▲" : dir === "desc" ? "▼" : ""}
                      </span>
                    </button>
                    {c.help && <HelpTip text={c.help} />}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, i) => (
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
                      className={`px-3 py-1.5 tabular-nums ${c.align === "right" ? "text-right" : "text-left"}`}
                      style={bg ? { backgroundColor: bg } : undefined}
                    >
                      {c.render ? c.render(row) : formatCell(raw)}
                    </td>
                  );
                })}
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-3 py-3 text-sm text-muted">
                  No rows match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatCell(v: unknown): React.ReactNode {
  if (v == null) return <span className="text-muted">n/a</span>;
  if (typeof v === "boolean") return v ? "✓" : "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(4);
  return String(v);
}

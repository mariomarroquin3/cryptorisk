export function fmtUsd(x: number | null | undefined, digits = 0): string {
  if (x == null || !Number.isFinite(x)) return "n/a";
  return x.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
  });
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x == null || !Number.isFinite(x)) return "n/a";
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtNum(x: number | null | undefined, digits = 4): string {
  if (x == null || !Number.isFinite(x)) return "n/a";
  return x.toFixed(digits);
}

export function fmtConfidence(alpha: number): string {
  return `${(100 * (1 - alpha)).toFixed(1)}%`;
}

export function fmtDate(s: string | null | undefined): string {
  if (!s) return "n/a";
  return s.slice(0, 10);
}

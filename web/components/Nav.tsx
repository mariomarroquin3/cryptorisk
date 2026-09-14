"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/models", label: "Model Comparison" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/capital", label: "Capital & Decision" },
  { href: "/regimes", label: "Regimes" },
];

/** Carried across pages so switching pages doesn't reset the user's asset
 * and confidence choice -- see lib/useQueryParam.ts. */
const PERSISTED_PARAMS = ["asset", "alpha"];

export function Nav() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const qs = PERSISTED_PARAMS
    .filter((k) => searchParams.get(k) != null)
    .map((k) => `${k}=${encodeURIComponent(searchParams.get(k) as string)}`)
    .join("&");
  return (
    <nav className="flex w-56 shrink-0 flex-col border-r border-grid bg-panel px-3 py-6">
      <Link href="/" className="mb-8 flex items-center gap-2 px-2">
        <span
          className="flex h-7 w-7 items-center justify-center rounded-md text-sm text-bg"
          style={{
            background: "linear-gradient(135deg, var(--amber), #ff7a1a)",
            boxShadow: "0 0 16px var(--glow-amber)",
          }}
        >
          &#9672;
        </span>
        <span className="font-display text-lg font-semibold tracking-wide text-text">CUBO+</span>
      </Link>
      <ul className="space-y-1">
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <li key={l.href}>
              <Link
                href={qs ? `${l.href}?${qs}` : l.href}
                className={`relative block rounded-md px-3 py-2 text-sm transition-colors ${
                  active
                    ? "bg-panel-2 text-text"
                    : "text-muted hover:bg-panel-2/60 hover:text-text"
                }`}
              >
                {active && (
                  <span
                    className="absolute inset-y-1 left-0 w-0.5 rounded-full"
                    style={{ background: "var(--amber)" }}
                  />
                )}
                {l.label}
              </Link>
            </li>
          );
        })}
      </ul>
      <div className="mt-auto px-2 pt-6 text-[0.65rem] text-muted/70">
        Comparative VaR/ES study &middot; BTC / ETH
      </div>
    </nav>
  );
}

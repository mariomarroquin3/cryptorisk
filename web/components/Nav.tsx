"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useState } from "react";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/models", label: "Model Comparison" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/capital", label: "Capital & Decision" },
  { href: "/regimes", label: "Regimes" },
  { href: "/live", label: "Live Record" },
  { href: "/explain", label: "Explainability" },
];

/** Carried across pages so switching pages doesn't reset the user's asset
 * and confidence choice -- see lib/useQueryParam.ts. */
const PERSISTED_PARAMS = ["asset", "alpha"];

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2 px-2">
      <span
        className="flex h-7 w-7 items-center justify-center rounded-md text-sm text-bg"
        style={{
          background: "linear-gradient(135deg, var(--amber), #ff7a1a)",
          boxShadow: "0 0 16px var(--glow-amber)",
        }}
      >
        &#9672;
      </span>
      <span className="font-display text-lg font-semibold tracking-wide text-text">Cryptorisk</span>
    </Link>
  );
}

export function Nav() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [open, setOpen] = useState(false);

  // Close the mobile drawer whenever the route changes (link click or
  // back/forward) -- the React-recommended "adjust state during render"
  // pattern in place of an effect, since this fires per navigation, not per
  // external subscription.
  const [prevPathname, setPrevPathname] = useState(pathname);
  if (pathname !== prevPathname) {
    setPrevPathname(pathname);
    setOpen(false);
  }

  const qs = PERSISTED_PARAMS
    .filter((k) => searchParams.get(k) != null)
    .map((k) => `${k}=${encodeURIComponent(searchParams.get(k) as string)}`)
    .join("&");

  const linkList = (
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
  );

  return (
    <>
      {/* Mobile top bar: hamburger + logo, hidden on md+ where the sidebar is static. */}
      <div className="flex items-center justify-between border-b border-grid bg-panel px-4 py-3 md:hidden">
        <Logo />
        <button
          type="button"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex h-8 w-8 items-center justify-center rounded-md border border-grid text-text hover:bg-panel-2"
        >
          <span className="sr-only">Toggle navigation</span>
          {open ? (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile drawer + backdrop */}
      {open && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <nav className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-grid bg-panel px-3 py-6">
            <div className="mb-8">
              <Logo />
            </div>
            {linkList}
            <div className="mt-auto px-2 pt-6 text-[0.65rem] text-muted/70">
              Comparative VaR/ES study &middot; BTC / ETH
            </div>
          </nav>
        </div>
      )}

      {/* Desktop sidebar */}
      <nav className="hidden w-56 shrink-0 flex-col border-r border-grid bg-panel px-3 py-6 md:flex">
        <div className="mb-8">
          <Logo />
        </div>
        {linkList}
        <div className="mt-auto px-2 pt-6 text-[0.65rem] text-muted/70">
          Comparative VaR/ES study &middot; BTC / ETH
        </div>
      </nav>
    </>
  );
}

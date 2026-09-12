"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/models", label: "Model Comparison" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/capital", label: "Capital & Decision" },
  { href: "/regimes", label: "Regimes" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="w-56 shrink-0 border-r border-grid bg-panel px-3 py-6">
      <div className="mb-8 px-2 text-lg font-semibold tracking-wide text-text">
        &#9672; CUBO+
      </div>
      <ul className="space-y-1">
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <li key={l.href}>
              <Link
                href={l.href}
                className={`block rounded px-3 py-2 text-sm transition-colors ${
                  active
                    ? "bg-grid text-text"
                    : "text-muted hover:bg-grid/60 hover:text-text"
                }`}
              >
                {l.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

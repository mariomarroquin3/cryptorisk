import { ModelInfo } from "@/lib/api";

/** Wraps a model name (e.g. a DataTable cell) with a hover/tap tooltip
 * showing its family, one-line idea, and known limitations -- so a new or
 * unfamiliar model name (like "Realized-SV") isn't an unexplained string in
 * a table. CSS-only, same group-hover/focus-within pattern as HelpTip, but
 * the trigger is the name itself (dotted underline) rather than a "?" icon,
 * since here the jargon *is* the visible text. */
export function ModelTip({ name, info }: { name: string; info: ModelInfo | undefined }) {
  if (!info) return <>{name}</>;
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        tabIndex={0}
        aria-label={`What is ${name}?`}
        className="cursor-help border-b border-dotted border-muted/60 text-left normal-case text-inherit hover:border-amber hover:text-amber focus:border-amber focus:text-amber focus:outline-none"
      >
        {name}
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-0 z-20 mb-1.5 w-72 rounded-md border border-grid bg-bg px-3 py-2.5 text-[0.7rem] leading-snug font-normal text-text normal-case opacity-0 shadow-lg transition-opacity duration-100 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        <div className="text-[0.6rem] font-semibold uppercase tracking-wide text-amber">
          {info.family}
        </div>
        <div className="mt-1">{info.idea}</div>
        {info.limitations.length > 0 && (
          <ul className="mt-1.5 space-y-0.5 border-t border-grid pt-1.5 text-muted">
            {info.limitations.map((l, i) => (
              <li key={i}>&middot; {l}</li>
            ))}
          </ul>
        )}
      </span>
    </span>
  );
}

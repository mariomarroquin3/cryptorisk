/** Small "?" affordance next to a jargon term. CSS-only tooltip (a
 * group-hover/focus pattern) -- no JS state, works with touch (tap focuses
 * the button) and keyboard (tab + focus-visible). */
export function HelpTip({ text }: { text: string }) {
  return (
    <span className="group relative ml-1 inline-flex align-middle">
      <button
        type="button"
        tabIndex={0}
        aria-label="What does this mean?"
        className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted/50 text-[0.55rem] leading-none text-muted normal-case hover:border-amber hover:text-amber focus:border-amber focus:text-amber focus:outline-none"
      >
        ?
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 w-56 -translate-x-1/2 rounded-md border border-grid bg-bg px-2.5 py-2 text-[0.7rem] leading-snug font-normal text-text normal-case opacity-0 shadow-lg transition-opacity duration-100 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}

import { createElement, Fragment, type ReactNode } from "react";

const ENTITIES: Record<string, string> = {
  "&mdash;": "—",
  "&ndash;": "–",
  "&amp;": "&",
};

function decodeEntities(text: string): string {
  return text.replace(/&(mdash|ndash|amp);/g, (m) => ENTITIES[m] ?? m);
}

/**
 * Minimal inline-markdown renderer for the study's own narrative bullets
 * (`study/run_portfolio.py::_read_bullets`), which use **bold**, `code`, and
 * a couple of HTML entities but nothing else. Not a general markdown parser.
 */
export function renderInlineMarkdown(text: string): ReactNode {
  const parts = decodeEntities(text.trim()).split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return createElement(
    Fragment,
    null,
    parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return createElement("strong", { key: i, className: "font-semibold text-text" }, part.slice(2, -2));
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return createElement(
          "code",
          { key: i, className: "rounded bg-panel-2 px-1 py-0.5 font-mono text-[0.85em]" },
          part.slice(1, -1),
        );
      }
      return part;
    }),
  );
}

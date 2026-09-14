"use client";

import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

/** A `useState`-shaped hook backed by a URL query param instead of component
 * state -- so a page's asset/confidence selection survives navigating to
 * another page (each page reads/writes the same param name) and links are
 * shareable. Uses `router.replace` (not `push`) so every selector change
 * doesn't pile up in browser history. */
export function useQueryParam(key: string, fallback: string | null = null) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const value = searchParams.get(key) ?? fallback;

  const setValue = useCallback(
    (next: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set(key, next);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [key, pathname, router, searchParams],
  );

  return [value, setValue] as const;
}

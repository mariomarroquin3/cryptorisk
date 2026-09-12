"use client";

import { useConfig } from "@/lib/hooks";
import { API_BASE } from "@/lib/api";

export function ApiStatusBanner() {
  const { error, isLoading } = useConfig();
  if (isLoading || !error) return null;
  return (
    <div className="border-b border-red bg-red/15 px-6 py-2 text-sm text-red">
      No se pudo conectar a la API en <span className="font-semibold">{API_BASE}</span>.
      &nbsp;Arráncala con <code className="rounded bg-panel px-1 py-0.5">make api</code> (o{" "}
      <code className="rounded bg-panel px-1 py-0.5">make dev</code>) y esta página se
      recuperará sola.
    </div>
  );
}

"use client";

import { useConfig } from "@/lib/hooks";
import { API_BASE } from "@/lib/api";

const IS_LOCAL = /^https?:\/\/(localhost|127\.0\.0\.1)/.test(API_BASE);

export function ApiStatusBanner() {
  const { error, isLoading } = useConfig();
  if (isLoading || !error) return null;
  return (
    <div className="border-b border-red/40 bg-red/10 px-6 py-2 text-sm text-red backdrop-blur-sm">
      No se pudo conectar a la API en <span className="font-semibold">{API_BASE}</span>.&nbsp;
      {IS_LOCAL ? (
        <>
          Arráncala con <code className="rounded bg-panel-2 px-1.5 py-0.5">make api</code> (o{" "}
          <code className="rounded bg-panel-2 px-1.5 py-0.5">make dev</code>) y esta página se
          recuperará sola.
        </>
      ) : (
        <>
          En el plan gratuito de Render la API se duerme si no hay tráfico y tarda hasta un minuto
          en despertar; esta página reintenta sola.
        </>
      )}
    </div>
  );
}

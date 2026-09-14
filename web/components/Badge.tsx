export function Badge({
  kind,
  children,
}: {
  kind: "live" | "frozen";
  children: React.ReactNode;
}) {
  const cls =
    kind === "live"
      ? "bg-green/10 text-green border-green/40"
      : "bg-blue/10 text-blue border-blue/40";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.65rem] font-semibold uppercase tracking-wide ${cls}`}
    >
      {kind === "live" && <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-green" />}
      {children}
    </span>
  );
}

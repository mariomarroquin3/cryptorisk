export function Badge({
  kind,
  children,
}: {
  kind: "live" | "frozen";
  children: React.ReactNode;
}) {
  const cls =
    kind === "live"
      ? "bg-green/15 text-green border-green"
      : "bg-blue/15 text-blue border-blue";
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide ${cls}`}
    >
      {children}
    </span>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-xs font-medium tracking-wide text-muted uppercase">
      {label}
      {children}
    </label>
  );
}

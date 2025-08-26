import * as React from "react";

/**
 * Minimal, dependency-free Select that mimics shadcn API:
 * <Select value onValueChange>
 *   <SelectTrigger><SelectValue placeholder="..." /></SelectTrigger>
 *   <SelectContent>
 *     <SelectItem value="a">A</SelectItem>
 *   </SelectContent>
 * </Select>
 *
 * Internally renders a native <select> for reliability.
 */

type SelectContextValue = {
  value?: string;
  placeholder?: string;
  onValueChange?: (v: string) => void;
  items: { value: string; label: React.ReactNode }[];
  setItems: React.Dispatch<React.SetStateAction<{ value: string; label: React.ReactNode }[]>>;
};

const SelectCtx = React.createContext<SelectContextValue | null>(null);

export function Select({
  value,
  defaultValue,
  onValueChange,
  children,
  className = "",
}: React.PropsWithChildren<{
  value?: string;
  defaultValue?: string;
  onValueChange?: (v: string) => void;
  className?: string;
}>) {
  const [val, setVal] = React.useState<string | undefined>(defaultValue);
  const [items, setItems] = React.useState<{ value: string; label: React.ReactNode }[]>([]);
  const controlled = value !== undefined;
  const current = controlled ? value : val;

  const ctx: SelectContextValue = {
    value: current,
    onValueChange: (v) => {
      if (!controlled) setVal(v);
      onValueChange?.(v);
    },
    items,
    setItems,
  } as any;

  return (
    <SelectCtx.Provider value={ctx}>
      <div className={className}>{children}</div>
    </SelectCtx.Provider>
  );
}

export function SelectTrigger({
  className = "",
  children,
}: React.PropsWithChildren<{ className?: string }>) {
  // purely presentational wrapper in this minimal build
  return <div className={`inline-flex items-center rounded-xl border px-2 h-8 ${className}`}>{children}</div>;
}

export function SelectValue({
  placeholder,
  className = "",
}: {
  placeholder?: string;
  className?: string;
}) {
  const ctx = React.useContext(SelectCtx);
  if (!ctx) return null;
  const label =
    ctx.items.find((i) => i.value === ctx.value)?.label ??
    (placeholder ? <span className="text-gray-400">{placeholder}</span> : null);
  return <div className={`text-sm ${className}`}>{label}</div>;
}

export function SelectContent({
  className = "",
  children,
}: React.PropsWithChildren<{ className?: string }>) {
  const ctx = React.useContext(SelectCtx);
  if (!ctx) return null;

  // Render a hidden native <select> that actually controls value
  return (
    <div className={`relative ${className}`}>
      <select
        className="absolute inset-0 opacity-0 cursor-pointer"
        value={ctx.value}
        onChange={(e) => ctx.onValueChange?.(e.target.value)}
        onBlur={() => { /* no-op */ }}
      >
        {ctx.items.map((i) => (
          <option key={i.value} value={i.value}>
            {String(i.value)}
          </option>
        ))}
      </select>

      {/* Visible list for clicks (optional) */}
      <div className="hidden">{children}</div>
    </div>
  );
}

export function SelectItem({
  value,
  children,
  className = "",
}: React.PropsWithChildren<{ value: string; className?: string }>) {
  const ctx = React.useContext(SelectCtx);
  React.useEffect(() => {
    if (ctx) ctx.setItems((prev) => (prev.some((i) => i.value === value) ? prev : [...prev, { value, label: children }]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // In this minimal version, SelectItem is not clickable itself; the native select above handles selection.
  return (
    <div className={`px-2 py-1 text-sm ${className}`} data-value={value}>
      {children}
    </div>
  );
}
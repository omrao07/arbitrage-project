// risk.tsx
import React, { useMemo, useRef, useState } from "react";

/**
 * Risk Registry — React + TypeScript + Tailwind
 * - Filter/search/sort a list of risks
 * - Add/Edit in a side panel
 * - CSV import/export
 * - Severity color & score (impact × likelihood)
 */

export type RiskStatus = "open" | "mitigating" | "closed";

export type RiskItem = {
  id: string;
  title: string;
  owner?: string;
  tag?: string;
  likelihood: number; // 1..5 (or your own scale)
  impact: number;     // 1..5
  notes?: string;
  status?: RiskStatus;
  createdAt?: string;
  updatedAt?: string;
};

export interface RiskProps {
  risks: RiskItem[];
  onChange?: (next: RiskItem[]) => void;
  /** Optional label sets for the numeric scales */
  likelihoodLabels?: string[]; // e.g. ["Rare","Unlikely","Possible","Likely","Almost certain"]
  impactLabels?: string[];     // e.g. ["Insignificant","Minor","Moderate","Major","Catastrophic"]
  title?: string;
}

type SortKey = "title" | "owner" | "tag" | "likelihood" | "impact" | "score" | "status" | "updatedAt";

const pill =
  "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] border border-zinc-700 bg-zinc-800/70";

const sevColor = (t: number) => {
  // t in [0..1]
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  // green -> yellow -> red
  const stops = [
    { t: 0, c: [0, 153, 74] },
    { t: 0.5, c: [255, 193, 7] },
    { t: 1, c: [239, 68, 68] },
  ];
  const a = stops.find((s) => t >= s.t) || stops[0];
  const b = stops.find((s) => s.t >= t) || stops[stops.length - 1];
  const u = (t - a.t) / Math.max(1e-9, b.t - a.t);
  const [r, g, bch] = [lerp(a.c[0], b.c[0], u), lerp(a.c[1], b.c[1], u), lerp(a.c[2], b.c[2], u)];
  return `rgb(${r}, ${g}, ${bch})`;
};

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

const Risk: React.FC<RiskProps> = ({
  risks,
  onChange,
  likelihoodLabels = ["1", "2", "3", "4", "5"],
  impactLabels = ["1", "2", "3", "4", "5"],
  title = "Risk Register",
}) => {
  const [q, setQ] = useState("");
  const [tag, setTag] = useState<string | "">("");
  const [status, setStatus] = useState<RiskStatus | "all">("all");
  const [sortBy, setSortBy] = useState<SortKey>("score");
  const [asc, setAsc] = useState(false);
  const [editing, setEditing] = useState<RiskItem | null>(null);

  const allTags = useMemo(() => Array.from(new Set(risks.map((r) => r.tag).filter(Boolean))) as string[], [risks]);

  const score = (r: RiskItem) => r.likelihood * r.impact;
  const maxScore = (likelihoodLabels.length || 5) * (impactLabels.length || 5);

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    return risks.filter((r) => {
      const mQ =
        !qq ||
        r.title.toLowerCase().includes(qq) ||
        (r.owner || "").toLowerCase().includes(qq) ||
        (r.tag || "").toLowerCase().includes(qq) ||
        (r.notes || "").toLowerCase().includes(qq);
      const mTag = !tag || r.tag === tag;
      const mStatus = status === "all" || r.status === status;
      return mQ && mTag && mStatus;
    });
  }, [risks, q, tag, status]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      let va: any, vb: any;
      switch (sortBy) {
        case "score":
          va = score(a);
          vb = score(b);
          break;
        default:
          va = (a as any)[sortBy];
          vb = (b as any)[sortBy];
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      return asc ? va - vb : vb - va;
    });
    return arr;
  }, [filtered, sortBy, asc]);

  const remove = (id: string) => {
    onChange?.(risks.filter((r) => r.id !== id));
  };

  const upsert = (r: RiskItem) => {
    const now = new Date().toISOString();
    const next = risks.some((x) => x.id === r.id)
      ? risks.map((x) => (x.id === r.id ? { ...r, updatedAt: now } : x))
      : [{ ...r, createdAt: now, updatedAt: now }, ...risks];
    onChange?.(next);
    setEditing(null);
  };

  const exportCSV = () => {
    const head = [
      "id",
      "title",
      "owner",
      "tag",
      "likelihood",
      "impact",
      "status",
      "notes",
      "createdAt",
      "updatedAt",
    ].join(",");
    const rows = risks.map((r) =>
      [
        r.id,
        `"${(r.title || "").replace(/"/g, '""')}"`,
        r.owner || "",
        r.tag || "",
        r.likelihood,
        r.impact,
        r.status || "",
        `"${(r.notes || "").replace(/"/g, '""')}"`,
        r.createdAt || "",
        r.updatedAt || "",
      ].join(",")
    );
    const blob = new Blob([head + "\n" + rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "risks.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const importRef = useRef<HTMLInputElement | null>(null);
  const importCSV = async (f: File) => {
    const txt = await f.text();
    const lines = txt.split(/\r?\n/).filter(Boolean);
    const [hdr, ...rest] = lines;
    const cols = hdr.split(",");
    const idx = (k: string) => cols.indexOf(k);
    const parse = (s: string) => s.replace(/^"|"$/g, "").replace(/""/g, '"');
    const parsed: RiskItem[] = rest.map((ln) => {
      const parts = ln.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || [];
      const get = (k: string) => parse(parts[idx(k)] || "");
      return {
        id: get("id") || Math.random().toString(36).slice(2),
        title: get("title"),
        owner: get("owner") || undefined,
        tag: get("tag") || undefined,
        likelihood: clamp(parseInt(get("likelihood") || "1", 10), 1, likelihoodLabels.length || 5),
        impact: clamp(parseInt(get("impact") || "1", 10), 1, impactLabels.length || 5),
        status: (get("status") as RiskStatus) || "open",
        notes: get("notes") || undefined,
        createdAt: get("createdAt") || undefined,
        updatedAt: get("updatedAt") || undefined,
      };
    });
    onChange?.(parsed);
  };

  const headerBtn = (k: SortKey, label: string) => (
    <button
      className="text-left w-full flex items-center gap-1"
      onClick={() => (setSortBy(k), setAsc(k === sortBy ? !asc : false))}
      title="Sort"
    >
      <span>{label}</span>
      <span className="text-zinc-500 text-xs">{sortBy === k ? (asc ? "▲" : "▼") : ""}</span>
    </button>
  );

  const Sev = ({ r }: { r: RiskItem }) => {
    const sc = score(r);
    const t = (sc - 1) / (maxScore - 1 || 1);
    return (
      <div className="flex items-center gap-2">
        <div
          className="h-2 w-8 rounded"
          style={{ background: `linear-gradient(90deg, ${sevColor(Math.max(0, t - 0.15))}, ${sevColor(t)})` }}
          title={`Score ${sc}`}
        />
        <span className="tabular-nums text-sm">{sc}</span>
      </div>
    );
  };

  return (
    <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-950">
      {/* Header + controls */}
      <div className="flex flex-wrap items-center gap-2 p-4 border-b border-zinc-800">
        <h2 className="text-zinc-100 font-semibold">{title}</h2>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search…"
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-amber-400/40 text-zinc-100"
          />
          <select
            value={tag}
            onChange={(e) => setTag(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-2 py-1.5 text-sm text-zinc-200"
          >
            <option value="">All tags</option>
            {allTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as any)}
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-2 py-1.5 text-sm text-zinc-200"
          >
            <option value="all">All status</option>
            <option value="open">Open</option>
            <option value="mitigating">Mitigating</option>
            <option value="closed">Closed</option>
          </select>
          <button
            onClick={() =>
              setEditing({
                id: Math.random().toString(36).slice(2),
                title: "",
                likelihood: 1,
                impact: 1,
                status: "open",
              })
            }
            className="px-3 py-1.5 rounded-lg bg-emerald-600/80 hover:bg-emerald-500 text-sm"
          >
            + New
          </button>
          <button onClick={exportCSV} className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-sm">
            Export CSV
          </button>
          <button
            onClick={() => importRef.current?.click()}
            className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-sm"
          >
            Import CSV
          </button>
          <input
            ref={importRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => e.target.files?.[0] && importCSV(e.target.files[0])}
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-900 text-zinc-300">
            <tr className="[&>th]:px-3 [&>th]:py-2 [&>th]:font-medium [&>th]:whitespace-nowrap">
              <th className="w-[28%]">{headerBtn("title", "Risk")}</th>
              <th className="w-[14%]">{headerBtn("owner", "Owner")}</th>
              <th className="w-[12%]">{headerBtn("tag", "Tag")}</th>
              <th className="w-[10%]">{headerBtn("likelihood", "Likelihood")}</th>
              <th className="w-[10%]">{headerBtn("impact", "Impact")}</th>
              <th className="w-[12%]">{headerBtn("score", "Score")}</th>
              <th className="w-[10%]">{headerBtn("status", "Status")}</th>
              <th className="w-[14%]">{headerBtn("updatedAt", "Updated")}</th>
              <th className="w-[10%] text-right pr-4">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800 text-zinc-200">
            {sorted.map((r) => (
              <tr key={r.id} className="[&>td]:px-3 [&>td]:py-2 align-top">
                <td>
                  <div className="font-medium">{r.title}</div>
                  {r.notes && <div className="text-zinc-400 text-xs line-clamp-2">{r.notes}</div>}
                </td>
                <td className="text-zinc-300">{r.owner || "—"}</td>
                <td>{r.tag ? <span className={pill}>{r.tag}</span> : "—"}</td>
                <td>
                  <div className="flex items-center gap-2">
                    <span className="tabular-nums">{r.likelihood}</span>
                    <span className="text-xs text-zinc-400">{likelihoodLabels[r.likelihood - 1]}</span>
                  </div>
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <span className="tabular-nums">{r.impact}</span>
                    <span className="text-xs text-zinc-400">{impactLabels[r.impact - 1]}</span>
                  </div>
                </td>
                <td><Sev r={r} /></td>
                <td>
                  <span
                    className={`${pill} ${
                      r.status === "open"
                        ? "bg-red-600/30 border-red-700/50"
                        : r.status === "mitigating"
                        ? "bg-amber-500/30 border-amber-600/50"
                        : "bg-emerald-600/30 border-emerald-700/50"
                    }`}
                  >
                    {r.status || "open"}
                  </span>
                </td>
                <td className="text-zinc-400">{r.updatedAt ? new Date(r.updatedAt).toLocaleString() : "—"}</td>
                <td className="text-right">
                  <div className="inline-flex gap-2">
                    <button
                      onClick={() => setEditing(r)}
                      className="px-2 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700 text-xs"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => remove(r.id)}
                      className="px-2 py-1 rounded-md bg-rose-600/80 hover:bg-rose-500 text-xs"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={9} className="text-center text-zinc-400 py-8">
                  No risks found. Try adjusting filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Editor Drawer */}
      {editing && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setEditing(null)}
            aria-hidden
          />
          <div className="absolute right-0 top-0 h-full w-full max-w-md bg-zinc-950 border-l border-zinc-800 shadow-2xl p-4 overflow-auto">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-zinc-100 font-semibold">{editing.title ? "Edit Risk" : "New Risk"}</h3>
              <button
                onClick={() => setEditing(null)}
                className="text-zinc-400 hover:text-zinc-200 text-sm"
              >
                Close
              </button>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                upsert(editing);
              }}
              className="space-y-3"
            >
              <label className="block">
                <div className="text-xs text-zinc-400 mb-1">Title</div>
                <input
                  value={editing.title}
                  onChange={(e) => setEditing({ ...editing, title: e.target.value })}
                  required
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 outline-none"
                />
              </label>

              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <div className="text-xs text-zinc-400 mb-1">Owner</div>
                  <input
                    value={editing.owner || ""}
                    onChange={(e) => setEditing({ ...editing, owner: e.target.value })}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 outline-none"
                  />
                </label>
                <label className="block">
                  <div className="text-xs text-zinc-400 mb-1">Tag</div>
                  <input
                    value={editing.tag || ""}
                    onChange={(e) => setEditing({ ...editing, tag: e.target.value })}
                    list="risk-tags"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 outline-none"
                  />
                  <datalist id="risk-tags">
                    {allTags.map((t) => (
                      <option key={t} value={t} />
                    ))}
                  </datalist>
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <div className="text-xs text-zinc-400 mb-1">Likelihood</div>
                  <select
                    value={editing.likelihood}
                    onChange={(e) =>
                      setEditing({ ...editing, likelihood: clamp(parseInt(e.target.value, 10), 1, likelihoodLabels.length || 5) })
                    }
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100"
                  >
                    {likelihoodLabels.map((lbl, i) => (
                      <option key={i} value={i + 1}>
                        {i + 1} — {lbl}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <div className="text-xs text-zinc-400 mb-1">Impact</div>
                  <select
                    value={editing.impact}
                    onChange={(e) =>
                      setEditing({ ...editing, impact: clamp(parseInt(e.target.value, 10), 1, impactLabels.length || 5) })
                    }
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100"
                  >
                    {impactLabels.map((lbl, i) => (
                      <option key={i} value={i + 1}>
                        {i + 1} — {lbl}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <label className="block">
                <div className="text-xs text-zinc-400 mb-1">Status</div>
                <select
                  value={editing.status || "open"}
                  onChange={(e) => setEditing({ ...editing, status: e.target.value as RiskStatus })}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100"
                >
                  <option value="open">Open</option>
                  <option value="mitigating">Mitigating</option>
                  <option value="closed">Closed</option>
                </select>
              </label>

              <label className="block">
                <div className="text-xs text-zinc-400 mb-1">Notes</div>
                <textarea
                  value={editing.notes || ""}
                  onChange={(e) => setEditing({ ...editing, notes: e.target.value })}
                  rows={4}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 outline-none"
                />
              </label>

              <div className="flex items-center justify-between pt-2">
                <div className="text-xs text-zinc-500">
                  Score: <span className="tabular-nums font-semibold">{editing.likelihood * editing.impact}</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setEditing(null)}
                    className="px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-sm"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-3 py-1.5 rounded-lg bg-emerald-600/80 hover:bg-emerald-500 text-sm"
                  >
                    Save
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Risk;
// app/components/householdswitcher.tsx
// No imports. No hooks. Self-contained household switcher with a <details> dropdown.
// - Click a household to switch (calls onSwitch)
// - Optional search, create, rename, delete (callbacks only; UI doesn’t keep its own state)
// - Pure inline styles; dark-mode friendly basics

"use client";

type Household = {
  id: string;
  name: string;
  members?: number | string[];   // count or list of names/emails
  color?: string;                 // avatar circle color
  avatarUrl?: string;             // optional logo/photo
  note?: string;                  // small subtitle
};

type Props = {
  households: Household[];
  currentId?: string;
  onSwitch: (id: string) => void | Promise<void>;
  allowSearch?: boolean;          // default true
  allowCreate?: boolean;          // default true
  onCreate?: (name: string) => void | string | { id: string } | Promise<void | string | { id: string }>;
  onRename?: (id: string, name: string) => void | Promise<void>;
  onDelete?: (id: string) => void | Promise<void>;
  label?: string;                 // summary label
};

export default function HouseholdSwitcher({
  households,
  currentId,
  onSwitch,
  allowSearch = true,
  allowCreate = true,
  onCreate,
  onRename,
  onDelete,
  label = "Household",
}: Props) {
  const cur = households.find((h) => h.id === currentId) || households[0];

  // ---- DOM helpers (no hooks) ----
  function filterList(e: any) {
    const q = (e.currentTarget as HTMLInputElement).value.toLowerCase();
    const list = (e.currentTarget as HTMLInputElement).closest("section")!.querySelector("#hsw-list")!;
    for (const li of Array.from(list.children) as HTMLElement[]) {
      const name = (li.getAttribute("data-name") || "").toLowerCase();
      li.style.display = name.includes(q) ? "" : "none";
    }
  }

  async function handleCreate(e: any) {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const input = form.querySelector('input[name="newName"]') as HTMLInputElement;
    const name = (input.value || "").trim();
    if (!name) return;
    try {
      const res = onCreate ? await onCreate(name) : undefined;
      const id = typeof res === "string" ? res : (res as any)?.id;
      if (id) await onSwitch(id);
      toast("Created");
      input.value = "";
      closeDropdown(form);
    } catch {
      toast("Create failed");
    }
  }

  async function doSwitch(id: string, el: HTMLElement) {
    try {
      await onSwitch(id);
      toast("Switched");
      closeDropdown(el);
    } catch {
      toast("Switch failed");
    }
  }

  async function doRename(id: string, el: HTMLElement) {
    if (!onRename) return;
    const name = prompt("New household name?");
    if (!name) return;
    try {
      await onRename(id, name.trim());
      toast("Renamed");
      closeDropdown(el);
    } catch {
      toast("Rename failed");
    }
  }

  async function doDelete(id: string, el: HTMLElement) {
    if (!onDelete) return;
    if (!confirm("Delete this household? This cannot be undone.")) return;
    try {
      await onDelete(id);
      toast("Deleted");
      closeDropdown(el);
    } catch {
      toast("Delete failed");
    }
  }

  function closeDropdown(node: Element) {
    const root = (node.closest("section") || document.body).querySelector("details#hsw") as HTMLDetailsElement | null;
    if (root) root.open = false;
  }

  function toast(msg: string) {
    const el = document.getElementById("hsw-toast");
    if (!el) return;
    el.textContent = msg;
    el.setAttribute("data-show", "1");
    setTimeout(() => el.removeAttribute("data-show"), 1200);
  }

  return (
    <section style={wrap} aria-label="Household switcher">
      <style>{css}</style>
      <div id="hsw-toast" style={toastStyle} />

      <details id="hsw" style={detailsBox}>
        <summary style={summary}>
          <div style={summaryLeft}>
            {renderAvatar(cur)}
            <div style={{ display: "grid" }}>
              <span style={summaryTitle}>{label}</span>
              <span style={summaryName}>{cur?.name || "Select…"}</span>
            </div>
          </div>
          <span aria-hidden="true" style={chev}>▾</span>
        </summary>

        <div style={panel}>
          {allowSearch ? (
            <div style={searchWrap}>
              <span style={searchIcon}>⌕</span>
              <input
                onInput={filterList}
                placeholder="Search households…"
                style={searchInput}
                aria-label="Search households"
              />
            </div>
          ) : null}

          <ul id="hsw-list" style={list}>
            {households.map((h) => (
              <li key={h.id} data-name={h.name} style={li}>
                <button
                  style={{ ...rowBtn, ...(h.id === cur?.id ? rowBtnActive : null) }}
                  onClick={(e) => doSwitch(h.id, e.currentTarget)}
                  aria-current={h.id === cur?.id ? "true" : "false"}
                >
                  {renderAvatar(h)}
                  <div style={{ display: "grid", textAlign: "left" }}>
                    <span style={rowName}>{h.name}</span>
                    <span style={rowMeta}>{fmtMembers(h.members)}{h.note ? ` · ${h.note}` : ""}</span>
                  </div>
                </button>

                {(onRename || onDelete) ? (
                  <div style={rowActions}>
                    {onRename ? (
                      <button title="Rename" style={iconBtn} onClick={(e) => doRename(h.id, e.currentTarget)}>✎</button>
                    ) : null}
                    {onDelete ? (
                      <button title="Delete" style={iconBtn} onClick={(e) => doDelete(h.id, e.currentTarget)}>🗑</button>
                    ) : null}
                  </div>
                ) : null}
              </li>
            ))}
          </ul>

          {allowCreate ? (
            <form onSubmit={handleCreate} style={createRow}>
              <input name="newName" placeholder="New household name" style={createInput} />
              <button type="submit" style={createBtn}>Create</button>
            </form>
          ) : null}
        </div>
      </details>
    </section>
  );
}

/* ---------------- small render helpers ---------------- */
function renderAvatar(h: Household) {
  const bg = h.color || pickColor(h.id || h.name || "");
  const initials = (h.name || "?")
    .split(/\s+/)
    .map((x) => x[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return h.avatarUrl ? (
    <img src={h.avatarUrl} alt="" width={28} height={28} style={avatarImg as any} />
  ) : (
    <span aria-hidden="true" style={{ ...avatar, background: bg }}>{initials}</span>
  );
}

function fmtMembers(m?: number | string[]) {
  if (Array.isArray(m)) return `${m.length} member${m.length === 1 ? "" : "s"}`;
  if (typeof m === "number") return `${m} member${m === 1 ? "" : "s"}`;
  return "—";
}

function pickColor(key: string) {
  const palette = ["#E0F2FE", "#FDE68A", "#DCFCE7", "#FCE7F3", "#E9D5FF", "#FFE4E6", "#FAE8FF", "#FFEFD5"];
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return palette[Math.abs(h) % palette.length];
}

/* ---------------- styles ---------------- */
const wrap: any = { position: "relative", display: "inline-block" };

const detailsBox: any = {
  border: "1px solid #e5e7eb",
  borderRadius: 12,
  background: "#fff",
  minWidth: 280,
};

const summary: any = {
  listStyle: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 8,
  padding: "8px 10px",
  cursor: "pointer",
  userSelect: "none",
};
const summaryLeft: any = { display: "flex", alignItems: "center", gap: 8 };
const summaryTitle: any = { fontSize: 11, color: "#6b7280" };
const summaryName: any = { fontWeight: 600, lineHeight: "18px" };
const chev: any = { fontSize: 12, color: "#6b7280" };

const panel: any = { padding: 10, borderTop: "1px solid #e5e7eb", display: "grid", gap: 8 };

const searchWrap: any = { position: "relative" };
const searchIcon: any = { position: "absolute", left: 8, top: 7, fontSize: 12, color: "#777" };
const searchInput: any = { width: "100%", height: 30, padding: "4px 8px 4px 24px", borderRadius: 8, border: "1px solid #e5e7eb", outline: "none" };

const list: any = { listStyle: "none", padding: 0, margin: 0, display: "grid", gap: 6, maxHeight: 280, overflow: "auto" };
const li: any = { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 };

const rowBtn: any = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  border: "1px solid #e5e7eb",
  background: "#fafafa",
  borderRadius: 10,
  padding: "6px 8px",
  cursor: "pointer",
  flex: 1,
  textAlign: "left",
};
const rowBtnActive: any = { background: "#111", borderColor: "#111", color: "#fff" };
const rowName: any = { fontSize: 13, fontWeight: 600 };
const rowMeta: any = { fontSize: 11.5, color: "inherit", opacity: 0.7 };

const rowActions: any = { display: "inline-flex", gap: 6, marginLeft: 4 };
const iconBtn: any = {
  border: "1px solid #e5e7eb",
  background: "#fff",
  borderRadius: 8,
  padding: "4px 6px",
  cursor: "pointer",
  fontSize: 12,
};

const createRow: any = { display: "flex", gap: 6, marginTop: 4 };
const createInput: any = { flex: 1, height: 30, padding: "6px 8px", borderRadius: 8, border: "1px solid #e5e7eb", outline: "none" };
const createBtn: any = { border: "1px solid #111", background: "#111", color: "#fff", borderRadius: 8, padding: "6px 10px", cursor: "pointer", fontSize: 13 };

const avatar: any = {
  width: 28,
  height: 28,
  borderRadius: "50%",
  display: "inline-grid",
  placeItems: "center",
  fontSize: 12,
  fontWeight: 700,
  color: "#111",
};
const avatarImg: any = {
  width: 28,
  height: 28,
  borderRadius: "50%",
  objectFit: "cover",
  background: "#f3f4f6",
};

const toastStyle: any = {
  position: "fixed",
  right: 16,
  bottom: 16,
  background: "#111",
  color: "#fff",
  padding: "8px 12px",
  borderRadius: 10,
  opacity: 0,
  transition: "opacity .25s ease",
  pointerEvents: "none",
  zIndex: 50,
};

const css = `
  details#hsw[open] { box-shadow: 0 8px 28px rgba(0,0,0,.08); }
  @media (prefers-color-scheme: dark) {
    details#hsw { background: #0b0b0c; border-color: rgba(255,255,255,0.1); }
    details#hsw summary { color: #fff; }
    details#hsw [style*="background: #fafafa"] { background: #111214 !important; border-color: rgba(255,255,255,0.1) !important; color: #e5e7eb !important; }
    input, button { color: inherit; }
    a { color: #9ecaff; }
    #hsw-toast[data-show="1"] { opacity: 1 !important; }
  }
`;

// xlsx/page.tsx
// Minimal, import-free page that lets you preview XLSX/CSV data in-browser
// without any external libs. It assumes your backend has an endpoint
// (e.g., /api/xlsx) that returns rows as JSON.

export default function XLSXPage() {
  return (
    <main className="min-h-screen bg-[#0b0f14] text-[#e6edf3] p-6">
      <header className="max-w-5xl mx-auto mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">XLSX Preview</h1>
        <p className="text-sm text-[#9fb0c0] mt-1">
          Lightweight table view for spreadsheet data.
        </p>
      </header>

      <section className="max-w-5xl mx-auto">
        <div className="overflow-x-auto border border-[#1f2b38] rounded-lg">
          <table className="min-w-full text-sm">
            <thead id="sheetHead" className="bg-[#111821] text-[#9fb0c0]"></thead>
            <tbody id="sheetBody"></tbody>
          </table>
        </div>
      </section>

      <footer className="max-w-5xl mx-auto mt-6 flex items-center gap-3">
        <button
          id="btnLoad"
          className="px-3 py-2 rounded-lg bg-[#1b2733] hover:bg-[#223142] border border-[#2b3b4a] text-sm"
        >
          Load Spreadsheet
        </button>
        <span id="status" className="text-xs text-[#9fb0c0]">Idle</span>
      </footer>

      <style >{`
        th, td { padding: 8px 12px; border-bottom: 1px solid #1f2b38; }
        th { font-weight: 500; text-align: left; }
        tr:hover td { background: rgba(255,255,255,0.03); }
      `}</style>

      <script
        // @ts-ignore
        dangerouslySetInnerHTML={{
          __html: `
(function(){
  const btn = document.getElementById("btnLoad");
  const status = document.getElementById("status");
  const head = document.getElementById("sheetHead");
  const body = document.getElementById("sheetBody");

  async function loadSheet() {
    status.textContent = "Loading…";
    try {
      // Replace /api/xlsx with your actual backend endpoint returning JSON
      const res = await fetch("/api/xlsx");
      const rows = await res.json();
      if (!Array.isArray(rows) || rows.length === 0) {
        status.textContent = "No data";
        return;
      }

      // headers
      const cols = Object.keys(rows[0]);
      head.innerHTML = "<tr>" + cols.map(c => "<th>"+c+"</th>").join("") + "</tr>";

      // rows
      body.innerHTML = "";
      rows.forEach(r => {
        const tr = document.createElement("tr");
        cols.forEach(c => {
          const td = document.createElement("td");
          td.textContent = r[c];
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });

      status.textContent = "Loaded " + rows.length + " rows";
    } catch (e) {
      status.textContent = "Error: " + e.message;
    }
  }

  if (btn) btn.addEventListener("click", loadSheet);
})();
          `,
        }}
      />
    </main>
  );
}
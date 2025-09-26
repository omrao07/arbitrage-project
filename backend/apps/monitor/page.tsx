// app/monitor/page.tsx
// Pure, import-free Next.js App Router page.
// Server Component that renders a self-contained monitoring dashboard
// with lightweight client-side behavior via inline <script> (no React hooks).

export default function MonitorPage() {
  return (
    <main className="min-h-screen bg-[#0b0f14] text-[#e6edf3] p-6">
      <header className="max-w-7xl mx-auto mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">System Monitor</h1>
        <p className="text-sm text-[#9fb0c0] mt-1">
          Live snapshot of services, data feeds, and risk metrics. No external imports.
        </p>
      </header>

      <section className="max-w-7xl mx-auto grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Uptime">
          <BigStat id="uptime" value="—" suffix="%" sub="rolling 24h" />
        </Card>
        <Card title="P50 Latency">
          <BigStat id="latencyP50" value="—" suffix=" ms" sub="HTTP/JSON API" />
        </Card>
        <Card title="Error Rate">
          <BigStat id="errorRate" value="—" suffix="%" sub="5xx per-minute" />
        </Card>
        <Card title="Queue Depth">
          <BigStat id="queueDepth" value="—" suffix="" sub="orders & jobs" />
        </Card>
      </section>

      <section className="max-w-7xl mx-auto grid gap-4 md:grid-cols-2 mt-6">
        <Card title="Data Feeds">
          <FeedTable />
        </Card>
        <Card title="Risk Snapshot">
          <RiskSnapshot />
        </Card>
      </section>

      <section className="max-w-7xl mx-auto mt-6 grid gap-4 lg:grid-cols-3">
        <Card title="Recent Alerts">
          <AlertsList />
        </Card>
        <Card title="Limits & Breaches">
          <LimitsPanel />
        </Card>
        <Card title="Last Payload">
          <Pre id="lastPayload" />
        </Card>
      </section>

      <footer className="max-w-7xl mx-auto mt-8 flex items-center gap-3">
        <button
          id="btnRefresh"
          className="px-3 py-2 rounded-lg bg-[#1b2733] hover:bg-[#223142] border border-[#2b3b4a] text-sm"
        >
          Refresh now
        </button>
        <label className="text-xs text-[#9fb0c0] flex items-center gap-2">
          <input id="autoToggle" type="checkbox" className="accent-[#7dd3fc]" defaultChecked />
          Auto-refresh (5s)
        </label>
        <span id="lastUpdated" className="ml-auto text-xs text-[#9fb0c0]">—</span>
      </footer>

      {/* Lightweight styles (no external CSS) */}
        <style>{`
        .card {
          background: radial-gradient(1200px 200px at -10% -20%, #0f1720 20%, #0b0f14 60%);
          border: 1px solid #203042;
          border-radius: 14px;
          padding: 16px;
        }
        .chip {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 2px 8px;
          border-radius: 999px;
          font-size: 11px;
          line-height: 16px;
          border: 1px solid #2a3b4b;
          background: #101720;
        }
        .ok { color: #86efac; border-color: #1f4b2e; background: #0d1b12; }
        .bad { color: #fda4af; border-color: #4b1f27; background: #1a0e11; }
        .warn { color: #fde68a; border-color: #4b3b1f; background: #1a160d; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px 8px; border-bottom: 1px solid #1b2631; font-size: 13px; }
        th { color: #9fb0c0; font-weight: 500; text-align: left; }
        tr:hover td { background: rgba(255,255,255,0.02); }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      `}</style>

      {/* Tiny client-side controller (no React hooks) */}
      <script
        // @ts-ignore
        dangerouslySetInnerHTML={{
          __html: `
(function () {
  const $ = (id) => document.getElementById(id);
  const fmt = {
    pct: (x) => (x==null? "—" : (x*100).toFixed(2) + "%"),
    ms: (x) => (x==null? "—" : Math.round(x).toString() + " ms"),
    int: (x) => (x==null? "—" : Intl.NumberFormat().format(Math.round(x))),
    num: (x, d=2) => (x==null? "—" : Number(x).toFixed(d))
  };

  function randN(mu, sigma) {
    let u=0, v=0;
    while(u===0) u=Math.random();
    while(v===0) v=Math.random();
    const z = Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
    return mu + sigma * z;
  }

  function synthSnapshot() {
    const uptime = Math.min(0.9999, Math.max(0.90, randN(0.997, 0.002)));
    const err = Math.max(0, randN(0.0025, 0.001));
    const p50 = Math.max(10, randN(42, 10));
    const qd  = Math.max(0, randN(128, 64));

    const feeds = [
      { key: "Equities",  latency: Math.abs(randN(120, 30)), status: Math.random()>0.03 ? "OK" : "DEGRADED" },
      { key: "Futures",   latency: Math.abs(randN(95, 20)),  status: Math.random()>0.02 ? "OK" : "DEGRADED" },
      { key: "FX",        latency: Math.abs(randN(60, 15)),  status: Math.random()>0.01 ? "OK" : "DOWN" },
      { key: "Crypto",    latency: Math.abs(randN(80, 25)),  status: Math.random()>0.05 ? "OK" : "DEGRADED" },
      { key: "News",      latency: Math.abs(randN(300, 80)), status: Math.random()>0.10 ? "OK" : "DELAYED" },
    ];

    const equity = Math.max(0, randN(12_500_000, 50_000));
    const peak   = 13_000_000;
    const dd     = peak > 0 ? (peak - equity)/peak : 0;
    const lev    = Math.max(0, randN(1.25, 0.15));
    const gross  = Math.max(0, randN(9_800_000, 200_000));
    const net    = Math.max(-2_000_000, Math.min(2_000_000, randN(250_000, 300_000)));
    const vol    = Math.max(0, randN(0.17, 0.03));
    const sharpe = randN(1.1, 0.3);

    const limits = [
      { name: "Max Drawdown", v: dd, limit: 0.20, type: "pct" },
      { name: "Max Leverage", v: lev, limit: 3.0,  type: "num" },
      { name: "Gross Exposure", v: gross, limit: 15_000_000, type: "usd" },
      { name: "Orders/min", v: Math.abs(randN(48, 12)), limit: 200, type: "num" },
    ];

    return {
      uptime, err, p50, qd, feeds,
      risk: { equity, dd, lev, gross, net, vol, sharpe, peak },
      limits,
      ts: new Date().toISOString()
    };
  }

  function setChip(el, stateText) {
    el.className = "chip " + (stateText==="OK" ? "ok" : (stateText==="DOWN" ? "bad":"warn"));
    el.textContent = stateText;
  }

  function render(snapshot) {
    const s = snapshot;
    const q = (id) => document.querySelector(id);

    if ($("uptime")) $("uptime").innerText = (s.uptime*100).toFixed(2);
    if ($("latencyP50")) $("latencyP50").innerText = Math.round(s.p50);
    if ($("errorRate")) $("errorRate").innerText = (s.err*100).toFixed(2);
    if ($("queueDepth")) $("queueDepth").innerText = (Math.round(s.qd)).toString();

    // feeds
    const body = q("#feedBody");
    if (body) {
      body.innerHTML = "";
      s.feeds.forEach((f) => {
        const tr = document.createElement("tr");
        const tdA = document.createElement("td");
        const tdB = document.createElement("td");
        const tdC = document.createElement("td");
        tdA.textContent = f.key;
        tdB.textContent = fmt.ms(f.latency);
        const chip = document.createElement("span");
        chip.className = "chip";
        setChip(chip, f.status);
        tdC.appendChild(chip);
        tr.appendChild(tdA); tr.appendChild(tdB); tr.appendChild(tdC);
        body.appendChild(tr);
      });
    }

    // risk
    const r = s.risk;
    if ($("riskEquity")) $("riskEquity").innerText = "$" + fmt.int(r.equity);
    if ($("riskDD")) $("riskDD").innerText = (r.dd*100).toFixed(2) + "%";
    if ($("riskLev")) $("riskLev").innerText = fmt.num(r.lev, 2) + "×";
    if ($("riskGross")) $("riskGross").innerText = "$" + fmt.int(r.gross);
    if ($("riskNet")) $("riskNet").innerText = "$" + fmt.int(r.net);
    if ($("riskVol")) $("riskVol").innerText = (r.vol*100).toFixed(2) + "%";
    if ($("riskSharpe")) $("riskSharpe").innerText = fmt.num(r.sharpe, 2);

    // limits
    const lim = q("#limitsBody");
    if (lim) {
      lim.innerHTML = "";
      s.limits.forEach((L) => {
        const tr = document.createElement("tr");
        function fmtVal(v, t) {
          if (t==="pct") return (v*100).toFixed(2) + "%";
          if (t==="usd") return "$" + fmt.int(v);
          return fmt.num(v, 2);
        }
        const td1 = document.createElement("td"); td1.textContent = L.name;
        const td2 = document.createElement("td"); td2.textContent = fmtVal(L.v, L.type);
        const td3 = document.createElement("td"); td3.textContent = fmtVal(L.limit, L.type);
        const td4 = document.createElement("td");
        const ok = (L.type==="pct" ? L.v <= L.limit
                  : L.type==="usd" ? L.v <= L.limit
                  : L.v <= L.limit);
        const chip = document.createElement("span");
        chip.className = "chip " + (ok ? "ok" : "bad");
        chip.textContent = ok ? "Within" : "Breach";
        td4.appendChild(chip);
        tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
        lim.appendChild(tr);
      });
    }

    // last payload
    const pre = $("lastPayload");
    if (pre) pre.textContent = JSON.stringify(s, null, 2);

    // timestamp
    const lu = $("lastUpdated");
    if (lu) lu.textContent = "Last updated: " + s.ts;
  }

  let timer = null;
  function start() {
    if (timer) clearInterval(timer);
    timer = setInterval(() => render(synthSnapshot()), 5000);
  }
  function stop() { if (timer) clearInterval(timer); timer = null; }

  document.addEventListener("click", (e) => {
    const t = e.target;
    if (t && t.id === "btnRefresh") render(synthSnapshot());
  });
  const chk = $("autoToggle");
  if (chk) {
    chk.addEventListener("change", (e) => {
      if (chk.checked) { start(); render(synthSnapshot()); }
      else { stop(); }
    });
  }

  // initial render
  render(synthSnapshot());
  if (chk && chk.checked) start();
})();
          `,
        }}
      />
    </main>
  );
}

/* ----------------------- Tiny, import-free subcomponents ----------------------- */

function Card(props: { title: string; children?: any }) {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-medium text-[#9fb0c0]">{props.title}</h2>
      </div>
      <div>{props.children}</div>
    </div>
  );
}

function BigStat(props: { id: string; value: string; suffix?: string; sub?: string }) {
  return (
    <div>
      <div className="flex items-end gap-2">
        <div id={props.id} className="text-3xl font-semibold tabular-nums">{props.value}</div>
        {props.suffix ? <div className="text-[#9fb0c0] mb-1">{props.suffix}</div> : null}
      </div>
      {props.sub ? <div className="text-xs text-[#9fb0c0] mt-1">{props.sub}</div> : null}
    </div>
  );
}

function FeedTable() {
  return (
    <div className="overflow-x-auto">
      <table>
        <thead>
          <tr>
            <th>Feed</th>
            <th>Latency</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="feedBody">
          <tr><td colSpan={3} className="text-[#9fb0c0]">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  );
}

function RiskSnapshot() {
  const Row = (p: { k: string; id: string }) => (
    <div className="flex items-center justify-between py-1">
      <span className="text-[#9fb0c0] text-sm">{p.k}</span>
      <span id={p.id} className="mono text-sm">—</span>
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-2">
      <Row k="Equity" id="riskEquity" />
      <Row k="Drawdown" id="riskDD" />
      <Row k="Leverage" id="riskLev" />
      <Row k="Gross" id="riskGross" />
      <Row k="Net" id="riskNet" />
      <Row k="Vol (ann.)" id="riskVol" />
      <Row k="Sharpe" id="riskSharpe" />
    </div>
  );
}

function AlertsList() {
  return (
    <ul id="alerts" className="space-y-2 text-sm">
      <li className="text-[#9fb0c0]">No critical alerts. Auto-refresh will populate synthetic alerts if any arise.</li>
    </ul>
  );
}

function LimitsPanel() {
  return (
    <div className="overflow-x-auto">
      <table>
        <thead>
          <tr>
            <th>Limit</th>
            <th>Value</th>
            <th>Max</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="limitsBody">
          <tr><td colSpan={4} className="text-[#9fb0c0]">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  );
}

function Pre(props: { id: string }) {
  return (
    <pre
      id={props.id}
      className="mono text-xs leading-relaxed whitespace-pre-wrap bg-[#0f1620] border border-[#203042] rounded-lg p-3 min-h-[180px]"
    >
      —
    </pre>
  );
}
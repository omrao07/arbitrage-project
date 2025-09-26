// src/main_gateway.cpp
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "md_snapshot.hpp"
#include "order_router.hpp"
#include "risk_wall.hpp"
#include "shared_memory.hpp"
#include "shm_layouts.hpp"

using clock_t = std::chrono::steady_clock;
static inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          clock_t::now().time_since_epoch()).count());
}

// ---------------- tiny config (key=value) ----------------
struct Config {
  std::string risk_shm = "/risk_wall";
  std::string md_ring  = "/md_ringbuf";
  std::string ord_in   = "/orders_in";
  std::string fills_out= "/fills_out";
  std::string hb_name  = "/heartbeat";
  uint32_t md_cap      = 1u << 15;  // 32768
  uint32_t ord_cap     = 1u << 12;  // 4096
  uint32_t fills_cap   = 1u << 12;  // 4096
  int heartbeat_ms     = 1000;
  bool demo_feed       = true;      // if false, read CSV lines from stdin
};

static Config load_config(const std::string& path) {
  Config c;
  std::ifstream f(path);
  if (!f) { std::cerr << "warn: cannot open config " << path << ", using defaults\n"; return c; }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0]=='#') continue;
    auto pos = line.find('=');
    if (pos==std::string::npos) continue;
    auto k = line.substr(0,pos); auto v = line.substr(pos+1);
    auto trim=[](std::string& s){ while(!s.empty() && isspace(s.back())) s.pop_back();
                                  size_t i=0; while(i<s.size() && isspace(s[i])) ++i; s.erase(0,i); };
    trim(k); trim(v);
    if (k=="risk_shm") c.risk_shm=v;
    else if (k=="md_ring") c.md_ring=v;
    else if (k=="orders_in") c.ord_in=v;
    else if (k=="fills_out") c.fills_out=v;
    else if (k=="heartbeat") c.hb_name=v;
    else if (k=="md_cap") c.md_cap=static_cast<uint32_t>(std::stoul(v));
    else if (k=="orders_cap") c.ord_cap=static_cast<uint32_t>(std::stoul(v));
    else if (k=="fills_cap") c.fills_cap=static_cast<uint32_t>(std::stoul(v));
    else if (k=="heartbeat_ms") c.heartbeat_ms=std::stoi(v);
    else if (k=="demo_feed") c.demo_feed=(v=="1"||v=="true");
  }
  return c;
}

// -------------- minimal SHM ring helper (SPSC) --------------
template <typename T>
class ShmRing {
public:
  static ShmRing open_or_create(const std::string& name, uint32_t capacity) {
    const size_t bytes = sizeof(shm_layout::RingBufHdr) + capacity * sizeof(T);
    auto reg = shm::MappedRegion::open_or_create(name, bytes, shm::Access::ReadWrite);
    auto* hdr = reinterpret_cast<shm_layout::RingBufHdr*>(reg.addr());
    if (hdr->capacity == 0 || hdr->elem_size != sizeof(T)) {
      std::memset(reg.addr(), 0, reg.size());
      hdr->head.store(0, std::memory_order_relaxed);
      hdr->tail.store(0, std::memory_order_relaxed);
      hdr->capacity = capacity;
      hdr->elem_size = sizeof(T);
    }
    auto* slots = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(reg.addr()) + sizeof(shm_layout::RingBufHdr));
    return ShmRing(std::move(reg), hdr, slots);
  }
  static ShmRing open_existing(const std::string& name, uint32_t capacity, shm::Access acc) {
    const size_t bytes = sizeof(shm_layout::RingBufHdr) + capacity * sizeof(T);
    auto reg = shm::MappedRegion::open_existing(name, bytes, acc);
    auto* hdr = reinterpret_cast<shm_layout::RingBufHdr*>(reg.addr());
    auto* slots = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(reg.addr()) + sizeof(shm_layout::RingBufHdr));
    return ShmRing(std::move(reg), hdr, slots);
  }
  bool push(const T& v) {
    const uint32_t cap = hdr_->capacity;
    const uint32_t head = hdr_->head.load(std::memory_order_relaxed);
    const uint32_t next = (head + 1) % cap;
    if (next == hdr_->tail.load(std::memory_order_acquire)) return false;
    slots_[head] = v;
    hdr_->head.store(next, std::memory_order_release);
    return true;
  }
  bool pop(T& out) {
    const uint32_t cap = hdr_->capacity;
    const uint32_t tail = hdr_->tail.load(std::memory_order_relaxed);
    if (tail == hdr_->head.load(std::memory_order_acquire)) return false;
    out = slots_[tail];
    hdr_->tail.store((tail + 1) % cap, std::memory_order_release);
    return true;
  }
private:
  ShmRing(shm::MappedRegion&& reg, shm_layout::RingBufHdr* hdr, T* slots)
    : reg_(std::move(reg)), hdr_(hdr), slots_(slots) {}
  shm::MappedRegion reg_;
  shm_layout::RingBufHdr* hdr_;
  T* slots_;
};

// -------------- global stop flag --------------
static std::atomic<bool> g_stop{false};
static void on_sig(int){ g_stop.store(true, std::memory_order_release); }

// -------------- main --------------
int main(int argc, char** argv) {
  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);

  // args: --config path (optional, INI-like key=value file as described above)
  Config cfg;
  for (int i=1;i<argc;i++){
    std::string a = argv[i];
    if (a=="--config" && i+1<argc) { cfg = load_config(argv[++i]); }
  }

  // ---- SHM bring-up ----
  auto md_rb    = ShmRing<shm_layout::QuoteMsg>::open_or_create(cfg.md_ring, cfg.md_cap);
  auto ord_rb   = ShmRing<shm_layout::OrderMsg>::open_or_create(cfg.ord_in, cfg.ord_cap);
  auto fills_rb = ShmRing<shm_layout::FillMsg>::open_or_create(cfg.fills_out, cfg.fills_cap);

  // Heartbeat
  auto hb_reg = shm::MappedRegion::open_or_create(cfg.hb_name, sizeof(shm_layout::Heartbeat), shm::Access::ReadWrite);
  auto* hb = reinterpret_cast<shm_layout::Heartbeat*>(hb_reg.addr());
  std::memset(hb, 0, sizeof(*hb));
  hb->alive.store(1, std::memory_order_release);
  hb->flags.store(0, std::memory_order_release);
  hb->ts_ns.store(now_ns(), std::memory_order_release);

  // Risk wall
  risk::RiskWall rw(cfg.risk_shm, /*bytes*/ 1<<20);
  risk::Limits lim; lim.max_gross_usd = 5e6; lim.max_symbol_pos = 50'000; lim.max_notional_usd = 2e6;
  rw.load_limits(lim);

  // Order TX sink (stdout stub)
  exec::OrderRouter router([&](const exec::Order& o){
    std::cout << "[TX] " << (o.side==exec::Side::Buy?"BUY ":"SELL ")
              << o.symbol << " qty=" << o.qty << " @ " << o.limit << "\n";
  });

  // ---- Threads ----

  // 1) Heartbeat updater
  std::thread th_hb([&]{
    while(!g_stop.load(std::memory_order_acquire)) {
      hb->ts_ns.store(now_ns(), std::memory_order_release);
      hb->alive.store(1, std::memory_order_release);
      std::this_thread::sleep_for(std::chrono::milliseconds(cfg.heartbeat_ms));
    }
    hb->alive.store(0, std::memory_order_release);
  });

  // 2) Orders consumer -> RiskWall precheck -> router -> fills_out ack
  std::thread th_ord([&]{
    shm_layout::OrderMsg om{};
    while(!g_stop.load(std::memory_order_acquire)) {
      bool any=false;
      while (ord_rb.pop(om)) {
        any = true;
        exec::Order o;
        o.symbol = std::string(om.sym, om.sym + strnlen(om.sym, sizeof(om.sym)));
        o.side   = (om.side==shm_layout::Side::Buy) ? exec::Side::Buy : exec::Side::Sell;
        o.qty    = om.qty;
        o.limit  = om.limit_px;
        o.ts_ns  = om.ts_ns;

        bool sent = router.route(o, [&](const exec::Order& ord, std::string& why){
          return rw.allow_order(ord.symbol, ord.limit, ord.qty, why);
        });

        shm_layout::FillMsg fm{};
        std::memset(&fm, 0, sizeof(fm));
        std::strncpy(fm.sym, o.symbol.c_str(), sizeof(fm.sym));
        fm.side = (o.side==exec::Side::Buy)? shm_layout::Side::Buy : shm_layout::Side::Sell;
        fm.qty  = o.qty;
        fm.px   = o.limit;
        fm.ts_ns= now_ns();
        fm.ack  = sent;
        (void)fills_rb.push(fm);

        if (sent) rw.on_fill(o.symbol, o.limit, o.qty);
      }
      if (!any) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  // 3) Market data ingestion → md_ringbuf
  // Build the snapshot; its callback pushes to md_ringbuf
  md::Snapshot snap([&](const std::string& sym, const md::Quote& q){
    shm_layout::QuoteMsg m{};
    std::memset(&m, 0, sizeof(m));
    std::strncpy(m.sym, sym.c_str(), sizeof(m.sym));
    m.bid = q.bid; m.ask = q.ask; m.ts_ns = q.ts_ns;
    (void)md_rb.push(m);
  });

  std::thread th_md([&]{
    if (cfg.demo_feed) {
      // tiny demo generator
      const char* syms[] = {"AAPL","MSFT","TSLA"};
      double base[] = {200.0, 410.0, 250.0};
      size_t n = sizeof(syms)/sizeof(syms[0]);
      size_t i=0;
      while(!g_stop.load(std::memory_order_acquire)) {
        std::string s = syms[i % n];
        double b = base[i % n];
        double k = ((i%11)-5) * 0.01;
        char line[128];
        std::snprintf(line, sizeof(line), "%s,%.2f,%.2f,%llu",
                      s.c_str(), b+k, b+k+0.02, (unsigned long long)now_ns());
        snap.ingest_line(line);
        ++i;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    } else {
      // read CSV lines "SYM,bid,ask,ts" from stdin
      std::string line;
      while(!g_stop.load(std::memory_order_acquire) && std::getline(std::cin, line)) {
        snap.ingest_line(line);
      }
    }
  });

  // ---- Main loop (just wait) ----
  std::cout << "hft_gateway running. Ctrl+C to stop.\n";
  while(!g_stop.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  th_md.join();
  th_ord.join();
  th_hb.join();
  std::cout << "hft_gateway stopped.\n";
  return 0;
}
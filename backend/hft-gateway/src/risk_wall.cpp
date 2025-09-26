// src/risk_wall.cpp
#include "risk_wall.hpp"
#include "shared_memory.hpp"

#include <cstring>   // std::memset
#include <cmath>     // std::abs
#include <mutex>
#include <stdexcept>

namespace risk {

struct RiskWall::Impl {
  Limits limits{};
  shm::MappedRegion reg;   // POSIX shm mapping
  State* st{nullptr};      // live state living in shm
  std::mutex mu;           // protects limits + map updates

  Impl(const std::string& shm_name, std::size_t bytes) {
    // Create or open the shared memory region and zero it on bring-up.
    reg = shm::MappedRegion::open_or_create(shm_name, bytes, shm::Access::ReadWrite);
    std::memset(reg.addr(), 0, reg.size());
    st = reinterpret_cast<State*>(reg.addr());
    // Initialize atomics explicitly (memset doesn't initialize atomics portably)
    st->gross_usd.store(0.0, std::memory_order_relaxed);
    st->realized_pnl.store(0.0, std::memory_order_relaxed);
    st->order_counter.store(0, std::memory_order_relaxed);
  }
};

RiskWall::RiskWall(const std::string& shm_name, std::size_t bytes)
  : p_(new Impl(shm_name, bytes)) {}

RiskWall::~RiskWall() {
  delete p_;
}

void RiskWall::load_limits(const Limits& l) {
  std::lock_guard<std::mutex> g(p_->mu);
  p_->limits = l;
}

bool RiskWall::allow_order(const std::string& symbol, double px, double qty, std::string& reason) {
  std::lock_guard<std::mutex> g(p_->mu);

  const double notional = std::abs(px * qty);
  if (p_->limits.max_notional_usd > 0.0 && notional > p_->limits.max_notional_usd) {
    reason = "per_order_notional";
    return false;
  }

  // Gross exposure check (very simple: sum of absolute notionals)
  const double gross_after = p_->st->gross_usd.load(std::memory_order_acquire) + notional;
  if (p_->limits.max_gross_usd > 0.0 && gross_after > p_->limits.max_gross_usd) {
    reason = "gross_limit";
    return false;
  }

  // Per-symbol position cap (toy implementation)
  double cur_pos = 0.0;
  auto it = p_->st->pos_by_symbol.find(symbol);
  if (it != p_->st->pos_by_symbol.end()) cur_pos = it->second;
  const double pos_after = cur_pos + qty;
  if (p_->limits.max_symbol_pos > 0.0 && std::abs(pos_after) > p_->limits.max_symbol_pos) {
    reason = "symbol_pos_limit";
    return false;
  }

  // (Optional) order-rate throttle would go here if you track timestamps/buckets.

  // Passed checks → bump counter
  p_->st->order_counter.fetch_add(1, std::memory_order_acq_rel);
  reason.clear();
  return true;
}

void RiskWall::on_fill(const std::string& symbol, double px, double qty) {
  std::lock_guard<std::mutex> g(p_->mu);

  const double notional = std::abs(px * qty);

  // Update gross notional (naive: add absolute notional on fills)
  const double cur_gross = p_->st->gross_usd.load(std::memory_order_acquire);
  p_->st->gross_usd.store(cur_gross + notional, std::memory_order_release);

  // Update per-symbol position
  p_->st->pos_by_symbol[symbol] += qty;
}

} // namespace risk
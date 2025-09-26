// include/risk_wall.hpp
#pragma once
// -----------------------------------------------------------------------------
// Risk Wall
// -----------------------------------------------------------------------------
// Central risk-checks: gross exposure, per-symbol limits, order throttles.
// Backed by a shared memory region so multiple processes (gateway, monitor)
// can see the same state.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace risk {

/// Risk limits (static, loaded from YAML)
struct Limits {
    double max_gross_usd{1e7};        ///< total gross notional cap
    double max_symbol_pos{100000};    ///< absolute per-symbol position cap
    double max_notional_usd{1e6};     ///< per-symbol notional cap
    double max_daily_loss_usd{1e5};   ///< daily stop-loss (if tracked)
    double max_order_rate_per_s{2000};///< throttle (orders per second)
};

/// Shared state stored in shm (dynamic, updated live)
struct State {
    std::atomic<double> gross_usd{0.0};            ///< current gross exposure
    std::atomic<double> realized_pnl{0.0};         ///< realized PnL
    std::unordered_map<std::string,double> pos_by_symbol; ///< live positions
    std::atomic<std::uint64_t> order_counter{0};   ///< orders sent (for throttle)
};

/// Risk wall class: wraps shm mapping + checks
class RiskWall {
public:
    /// Create/open a risk wall over a shm region
    RiskWall(const std::string& shm_name, std::size_t bytes);

    ~RiskWall();

    /// Load static limits from YAML/config
    void load_limits(const Limits& l);

    /// Check if an order is allowed
    /// @param symbol instrument
    /// @param px     price
    /// @param qty    quantity
    /// @param reason string filled if rejected
    /// @return true if allowed, false if blocked
    bool allow_order(const std::string& symbol,
                     double px,
                     double qty,
                     std::string& reason);

    /// Update state on fill (positions + gross)
    void on_fill(const std::string& symbol, double px, double qty);

private:
    struct Impl;
    Impl* p_;
};

} // namespace risk
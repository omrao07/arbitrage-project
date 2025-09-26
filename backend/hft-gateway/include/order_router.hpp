// include/order_router.hpp
#pragma once
// -----------------------------------------------------------------------------
// Order Router
// -----------------------------------------------------------------------------
// Provides a simple interface to construct orders and route them to an
// execution sink (FPGA, FIX engine, stdout). Each route() call may perform
// pre-checks (e.g. risk wall, throttle) before transmission.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <functional>

namespace exec {

/// Side enum for buy/sell
enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

/// Simple order representation
struct Order {
    std::string symbol;      ///< instrument symbol (e.g. "AAPL")
    Side side{Side::Buy};    ///< order side
    double qty{0.0};         ///< quantity (shares, contracts)
    double limit{0.0};       ///< limit price
    std::uint64_t ts_ns{0};  ///< client timestamp (ns)
};

/// Transmit function type (to exchange/FIX/FPGA)
using TxFn = std::function<void(const Order&)>;

/// Router class: applies pre-checks then transmits
class OrderRouter {
public:
    /// Construct with a transmit function (exchange adapter)
    explicit OrderRouter(TxFn tx);

    /// Attempt to route an order.
    /// @param o   Order to be routed
    /// @param precheck  A function that validates the order; should return true
    ///                  if allowed, false otherwise. It may fill 'reason' string.
    /// @return true if order was transmitted, false if rejected.
    bool route(const Order& o,
               const std::function<bool(const Order&, std::string&)>& precheck);

private:
    TxFn tx_;
};

} // namespace exec
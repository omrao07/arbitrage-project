// src/order_router.cpp
#include "order_router.hpp"

namespace exec {

OrderRouter::OrderRouter(TxFn tx) : tx_(std::move(tx)) {}

bool OrderRouter::route(
    const Order& o,
    const std::function<bool(const Order&, std::string&)>& precheck
) {
    std::string reason;
    if (precheck) {
        if (!precheck(o, reason)) {
            // Rejected by precheck (risk/throttle/etc.)
            return false;
        }
    }
    // Passed checks → transmit
    if (tx_) tx_(o);
    return true;
}

} // namespace exec
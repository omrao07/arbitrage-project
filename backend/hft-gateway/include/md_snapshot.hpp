// include/md_snapshot.hpp
#pragma once
// -----------------------------------------------------------------------------
// Market Data Snapshot Handler
// -----------------------------------------------------------------------------
// Provides a simple interface to ingest raw feed lines (CSV/ITCH adapter can
// convert to text form) and dispatch structured Quote updates to a callback.
// In production you’d replace the parser with a proper binary feed decoder.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>

namespace md {

/// Single quote snapshot (best bid/ask)
struct Quote {
    double bid{0.0};
    double ask{0.0};
    std::uint64_t ts_ns{0};  ///< source timestamp (nanoseconds since epoch)
};

/// Callback type for quote updates
using OnQuote = std::function<void(const std::string&, const Quote&)>;

/// Market Data Snapshot store/handler
class Snapshot {
public:
    /// Construct with callback that will be invoked on every new quote
    explicit Snapshot(OnQuote cb);

    /// Ingest a raw feed line in the form "SYM,bid,ask,ts"
    /// Example: "AAPL,199.95,200.05,1705000000123456789"
    void ingest_line(const std::string& line);

    /// Retrieve latest quote for a symbol (thread-safe).
    /// Returns true if found, false otherwise.
    bool get_quote(const std::string& sym, Quote& out) const;

    /// Number of symbols currently tracked
    std::size_t size() const;

private:
    OnQuote cb_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Quote> book_;
};

} // namespace md
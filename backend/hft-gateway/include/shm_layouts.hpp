// include/shm_layouts.hpp
#pragma once
// -----------------------------------------------------------------------------
// Shared Memory Layouts
// -----------------------------------------------------------------------------
// Defines POD structs and headers for the different shared memory regions used
// in the gateway:
//   • /risk_wall   : risk state (gross, PnL, positions summary)
//   • /md_ringbuf  : quotes ring buffer (producer = feed decoder, consumer = strat)
//   • /orders_in   : inbound orders (producer = strategy, consumer = gateway)
//   • /fills_out   : outbound fills/acks (producer = gateway, consumer = strategy)
//   • /heartbeat   : gateway liveness/health
//
// These structs are laid out explicitly so both C++ and Python clients can mmap
// them consistently. Avoid STL containers here; use fixed arrays or offsets.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <atomic>

namespace shm_layout {

// ---------------- Risk Wall ----------------
/// Risk wall live state (minimal)
struct RiskWall {
    std::atomic<double> gross_usd;      ///< current gross exposure
    std::atomic<double> realized_pnl;   ///< realized PnL
    std::atomic<uint64_t> order_counter;///< total orders routed
    // For simplicity, per-symbol positions are not embedded here because
    // std::unordered_map is not POD. Use a separate ring buffer or external store.
};

// ---------------- Generic Ring Buffer Header ----------------
/// Ring buffer header for any message type
struct RingBufHdr {
    std::atomic<uint32_t> head;
    std::atomic<uint32_t> tail;
    uint32_t capacity;   ///< number of slots
    uint32_t elem_size;  ///< bytes per element
};

// ---------------- Market Data Message ----------------
struct QuoteMsg {
    char     sym[8];     ///< symbol (null-padded)
    double   bid;
    double   ask;
    uint64_t ts_ns;
};

// ---------------- Order Message ----------------
enum class Side : uint8_t { Buy=0, Sell=1 };

struct OrderMsg {
    char     sym[8];     ///< symbol (null-padded)
    Side     side;
    double   qty;
    double   limit_px;
    uint64_t ts_ns;      ///< client ts
};

// ---------------- Fill / Ack Message ----------------
struct FillMsg {
    char     sym[8];
    Side     side;
    double   qty;
    double   px;
    uint64_t ts_ns;      ///< gateway/exchange ts
    bool     ack;        ///< true if accepted, false if rejected
};

// ---------------- Heartbeat ----------------
struct Heartbeat {
    std::atomic<uint64_t> ts_ns;    ///< last updated timestamp
    std::atomic<uint32_t> alive;    ///< 1=healthy, 0=down
    std::atomic<uint32_t> flags;    ///< extra status bits
};

} // namespace shm_layout
// include/ringbuf.hpp
#pragma once
// -----------------------------------------------------------------------------
// Lock-free SPSC (single producer, single consumer) ring buffer
// -----------------------------------------------------------------------------
// Designed for shared memory (mmap'ed region): one writer thread/process and
// one reader thread/process. Head and tail are atomics to allow safe progress.
// -----------------------------------------------------------------------------
//
// Usage example:
//
//   struct Msg { uint64_t ts; double px; double qty; };
//   constexpr size_t CAP = 1024;
//   shm::Region reg = shm::open_or_create("/orders_in", sizeof(ringbuf::Header)+CAP*sizeof(Msg));
//   auto* hdr = new(reg.addr) ringbuf::Header(CAP, sizeof(Msg));
//   auto* rb  = reinterpret_cast<ringbuf::RingBuf<Msg>*>(reg.addr);
//   rb->push(msg);  // producer
//   Msg out; if (rb->pop(out)) { ... }  // consumer
//
// -----------------------------------------------------------------------------

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ringbuf {

/// Shared memory ring buffer header
struct Header {
    std::atomic<uint32_t> head;
    std::atomic<uint32_t> tail;
    uint32_t capacity;
    uint32_t elem_size;

    Header(uint32_t cap, uint32_t esz)
        : head(0), tail(0), capacity(cap), elem_size(esz) {}
};

/// Ring buffer template operating on plain structs (POD)
template <typename T>
struct RingBuf {
    Header hdr;
    T      slots[1];  // actually [capacity], flexible array trick

    static size_t bytes_for(uint32_t capacity) {
        return sizeof(Header) + capacity * sizeof(T);
    }

    bool push(const T& v) {
        uint32_t cap  = hdr.capacity;
        uint32_t head = hdr.head.load(std::memory_order_relaxed);
        uint32_t next = (head + 1) % cap;

        if (next == hdr.tail.load(std::memory_order_acquire)) {
            return false;  // full
        }
        slots[head] = v;
        hdr.head.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        uint32_t cap  = hdr.capacity;
        uint32_t tail = hdr.tail.load(std::memory_order_relaxed);

        if (tail == hdr.head.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        out = slots[tail];
        uint32_t next = (tail + 1) % cap;
        hdr.tail.store(next, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return hdr.head.load(std::memory_order_acquire) ==
               hdr.tail.load(std::memory_order_acquire);
    }

    bool full() const {
        uint32_t cap  = hdr.capacity;
        uint32_t head = hdr.head.load(std::memory_order_acquire);
        uint32_t next = (head + 1) % cap;
        return next == hdr.tail.load(std::memory_order_acquire);
    }

    uint32_t size() const {
        int32_t h = hdr.head.load(std::memory_order_acquire);
        int32_t t = hdr.tail.load(std::memory_order_acquire);
        if (h >= t) return h - t;
        return hdr.capacity - (t - h);
    }
};

} // namespace ringbuf
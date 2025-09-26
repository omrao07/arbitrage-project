#!/usr/bin/env python3
"""
scripts/perf_bench.py
-----------------------------------------------------------
Benchmark the ring buffer in shared memory.
Measures push/pop latency and throughput for synthetic messages.
"""

import mmap
import os
import struct
import sys
import time
import statistics

# Parameters
N_MSGS = 100_000
SYM    = b"AAPL\0\0\0\0"  # 8 bytes symbol
SHM_NAME = "/perf_ringbuf"
CAPACITY = 1024
MSG_FMT = "<8s d d Q"     # sym[8], bid(double), ask(double), ts(uint64)
MSG_SIZE = struct.calcsize(MSG_FMT)


def create_shm(name: str, size: int):
    """Open or create a POSIX shm region and resize."""
    fd = os.open(f"/dev/shm{name}", os.O_CREAT | os.O_RDWR)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    os.close(fd)
    return mm


def run_bench():
    # Layout: head(uint32), tail(uint32), capacity(uint32), elem_size(uint32), then slots
    HDR_FMT = "<IIII"
    HDR_SIZE = struct.calcsize(HDR_FMT)
    shm_size = HDR_SIZE + CAPACITY * MSG_SIZE
    mm = create_shm(SHM_NAME, shm_size)

    # initialize header
    mm.seek(0)
    mm.write(struct.pack(HDR_FMT, 0, 0, CAPACITY, MSG_SIZE))
    mm.flush()

    head = tail = 0
    latencies = []

    for i in range(N_MSGS):
        # Encode message
        bid = 100.0 + (i % 50) * 0.01
        ask = bid + 0.02
        ts  = time.time_ns()
        msg = struct.pack(MSG_FMT, SYM, bid, ask, ts)

        t0 = time.perf_counter_ns()

        # push
        pos = HDR_SIZE + (head % CAPACITY) * MSG_SIZE
        mm.seek(pos)
        mm.write(msg)
        head = (head + 1) % CAPACITY

        # pop
        pos = HDR_SIZE + (tail % CAPACITY) * MSG_SIZE
        mm.seek(pos)
        data = mm.read(MSG_SIZE)
        tail = (tail + 1) % CAPACITY

        t1 = time.perf_counter_ns()
        latencies.append(t1 - t0)

    mm.close()

    # stats
    med = statistics.median(latencies)
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    print(f"Msgs: {N_MSGS}")
    print(f"Median latency: {med:.1f} ns")
    print(f"99th pct latency: {p99:.1f} ns")
    print(f"Throughput: {N_MSGS / (sum(latencies)/1e9):.0f} msg/s")


if __name__ == "__main__":
    run_bench()
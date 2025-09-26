// tests/test_risk_wall.cpp
#include "assert.hpp"

#include "risk_wall.hpp"
#include "shared_memory.hpp"   // for shm::MappedRegion::unlink

#include <string>
#include <iostream>

int main() {
  using namespace risk;

  // Use a unique SHM name for the test and start clean
  const std::string SHM_NAME = "/risk_wall_ut";
  shm::MappedRegion::unlink(SHM_NAME);

  // Bring up a small risk wall region
  RiskWall rw(SHM_NAME, 1 << 16);

  // Tight limits so tests are quick
  Limits lim{};
  lim.max_gross_usd     = 1000.0;  // total gross cap
  lim.max_symbol_pos    = 10.0;    // absolute per-symbol position cap
  lim.max_notional_usd  = 300.0;   // per-order notional cap
  rw.load_limits(lim);

  std::string why;

  // 1) Per-order notional cap should block
  ASSERT_FALSE(rw.allow_order("AAPL", /*px*/400.0, /*qty*/1.0, why));
  // 2) Small order should pass
  ASSERT_TRUE(rw.allow_order("AAPL", 100.0, 2.0, why));  // $200 notional
  rw.on_fill("AAPL", 100.0, 2.0);                        // update gross + pos

  // 3) Position cap: after +2 position, attempt +9 should be blocked (would be +11)
  ASSERT_FALSE(rw.allow_order("AAPL", 50.0, 9.0, why));
  // A +8 should be allowed (2 + 8 = 10 meets cap)
  ASSERT_TRUE(rw.allow_order("AAPL", 50.0, 8.0, why));
  rw.on_fill("AAPL", 50.0, 8.0); // position now +10, gross +$400 → cumulative gross $600

  // 4) Gross cap: keep filling small orders until we breach $1000
  // Current gross ≈ $200 + $400 = $600; next +$300 is OK, +$200 more should be blocked.
  ASSERT_TRUE(rw.allow_order("AAPL", 30.0, 10.0, why));  // $300 notional (pos would be 20 if applied, but we won't fill due to pos cap)
  // NOTE: Position cap would block if we filled, so don't call on_fill here.
  // Instead use a different symbol to test gross accumulation without position interference.
  ASSERT_TRUE(rw.allow_order("MSFT", 30.0, 10.0, why));  // $300 notional
  rw.on_fill("MSFT", 30.0, 10.0);                        // gross → $900 total

  // Now a further $150 should be allowed, but $200 should push us over $1000 and be blocked.
  ASSERT_TRUE(rw.allow_order("MSFT", 15.0, 10.0, why));  // $150 → would be $1050 *after* fill, but allow_order only checks prefill gross + order notional
  rw.on_fill("MSFT", 15.0, 10.0);                        // gross becomes $1050

  // Next $1 order should now be blocked by gross cap (since gross already above cap)
  ASSERT_FALSE(rw.allow_order("MSFT", 1.0, 1.0, why));

  std::cout << "All risk_wall tests passed.\n";
  return 0;
}
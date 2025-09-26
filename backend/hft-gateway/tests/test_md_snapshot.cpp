// tests/test_md_snapshot.cpp
#include "assert.hpp"

#include "md_snapshot.hpp"

#include <iostream>
#include <vector>
#include <string>
#include "_assert.hpp"

int main() {
  using namespace md;
  std::vector<std::pair<std::string, Quote>> seen;

  // Callback pushes into seen vector
  Snapshot snap([&](const std::string& sym, const Quote& q) {
    seen.emplace_back(sym, q);
  });

  // --- Valid line with timestamp ---
  snap.ingest_line("AAPL,199.95,200.05,123456789");
  ASSERT_EQ(seen.size(), 1u);
  ASSERT_STR_EQ(seen[0].first, "AAPL");
  ASSERT_NEAR(seen[0].second.bid, 199.95, 1e-6);
  ASSERT_NEAR(seen[0].second.ask, 200.05, 1e-6);
  ASSERT_EQ(seen[0].second.ts_ns, 123456789ull);

  // --- Valid line without timestamp ---
  snap.ingest_line("MSFT,410.1,410.2");
  ASSERT_EQ(seen.size(), 2u);
  ASSERT_STR_EQ(seen.back().first, "MSFT");
  ASSERT_NEAR(seen.back().second.bid, 410.1, 1e-6);
  ASSERT_NEAR(seen.back().second.ask, 410.2, 1e-6);
  ASSERT_EQ(seen.back().second.ts_ns, 0ull);

  // --- Extra whitespace should be trimmed ---
  snap.ingest_line(" TSLA , 250.5 , 251.0 , 42 ");
  ASSERT_EQ(seen.size(), 3u);
  ASSERT_STR_EQ(seen.back().first, "TSLA");
  ASSERT_NEAR(seen.back().second.bid, 250.5, 1e-6);
  ASSERT_NEAR(seen.back().second.ask, 251.0, 1e-6);
  ASSERT_EQ(seen.back().second.ts_ns, 42ull);

  // --- Malformed line should be ignored ---
  snap.ingest_line("INVALID_LINE");
  ASSERT_EQ(seen.size(), 3u);  // unchanged

  // --- Verify get_quote and size ---
  Quote out{};
  ASSERT_TRUE(snap.get_quote("AAPL", out));
  ASSERT_NEAR(out.bid, 199.95, 1e-6);
  ASSERT_EQ(snap.size(), 3u);

  ASSERT_FALSE(snap.get_quote("NOPE", out));

  std::cout << "All md_snapshot tests passed.\n";
  return 0;
}
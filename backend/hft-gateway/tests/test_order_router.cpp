// tests/test_order_router.cpp
#include "assert.hpp"
#include "order_router.hpp"

#include <vector>
#include <string>
#include <iostream>

using exec::Order;
using exec::OrderRouter;
using exec::Side;

int main() {
  // capture-sink for transmitted orders
  std::vector<Order> sent;
  OrderRouter router([&](const Order& o){ sent.push_back(o); });

  // --- 1) Rejection path: precheck returns false ---
  Order o1{"AAPL", Side::Buy, 100.0, 200.00, 111};
  bool ok = router.route(o1, [](const Order& /*o*/, std::string& why){
    why = "blocked_for_test"; return false;
  });
  ASSERT_FALSE(ok);
  ASSERT_EQ(sent.size(), 0u);  // nothing transmitted

  // --- 2) Accept path: precheck returns true ---
  Order o2{"MSFT", Side::Sell, 50.0, 410.25, 222};
  ok = router.route(o2, [](const Order& /*o*/, std::string& why){
    why.clear(); return true;
  });
  ASSERT_TRUE(ok);
  ASSERT_EQ(sent.size(), 1u);
  ASSERT_STR_EQ(sent[0].symbol, "MSFT");
  ASSERT_EQ(static_cast<int>(sent[0].side), static_cast<int>(Side::Sell));
  ASSERT_NEAR(sent[0].qty, 50.0, 1e-9);
  ASSERT_NEAR(sent[0].limit, 410.25, 1e-9);
  ASSERT_EQ(sent[0].ts_ns, 222ull);

  // --- 3) No precheck provided: should transmit ---
  Order o3{"TSLA", Side::Buy, 10.0, 250.10, 333};
  ok = router.route(o3, nullptr);
  ASSERT_TRUE(ok);
  ASSERT_EQ(sent.size(), 2u);
  ASSERT_STR_EQ(sent[1].symbol, "TSLA");

  // --- 4) Multiple routes to ensure accumulation ---
  for (int i = 0; i < 3; ++i) {
    Order oi{"AAPL", Side::Buy, 1.0 + i, 200.0 + i, 1000 + static_cast<unsigned>(i)};
    ok = router.route(oi, [](const Order&, std::string&){ return true; });
    ASSERT_TRUE(ok);
  }
  ASSERT_EQ(sent.size(), 5u);

  std::cout << "All order_router tests passed.\n";
  return 0;
}
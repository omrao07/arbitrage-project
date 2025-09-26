// tests/assert.hpp
#pragma once
// -----------------------------------------------------------------------------
// Minimal test assertions for unit tests
// -----------------------------------------------------------------------------
// Usage:
//   #include "assert.hpp"
//
//   int main() {
//     int x = 2+2;
//     ASSERT_EQ(x, 4);
//     ASSERT_TRUE(x == 4);
//     ASSERT_NEAR(3.1415, 3.14, 0.01);
//     return 0;
//   }
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#define ASSERT_TRUE(cond) \
  do { \
    if (!(cond)) { \
      std::cerr << "[ASSERT_TRUE failed] " << __FILE__ << ":" << __LINE__ \
                << " : " #cond "\n"; \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a,b) \
  do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va == _vb)) { \
      std::cerr << "[ASSERT_EQ failed] " << __FILE__ << ":" << __LINE__ \
                << " : (" #a " == " #b ") " \
                << "LHS=" << _va << " RHS=" << _vb << "\n"; \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_NE(a,b) \
  do { \
    auto _va = (a); auto _vb = (b); \
    if (!(_va != _vb)) { \
      std::cerr << "[ASSERT_NE failed] " << __FILE__ << ":" << __LINE__ \
                << " : (" #a " != " #b ") both=" << _va << "\n"; \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_NEAR(a,b,eps) \
  do { \
    auto _va = (a); auto _vb = (b); auto _eps = (eps); \
    if (std::fabs(_va - _vb) > _eps) { \
      std::cerr << "[ASSERT_NEAR failed] " << __FILE__ << ":" << __LINE__ \
                << " : (" #a " ≈ " #b " ±" #eps ") " \
                << "LHS=" << _va << " RHS=" << _vb << " diff=" << std::fabs(_va - _vb) << "\n"; \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_STR_EQ(a,b) \
  do { \
    std::string _sa(a); std::string _sb(b); \
    if (!(_sa == _sb)) { \
      std::cerr << "[ASSERT_STR_EQ failed] " << __FILE__ << ":" << __LINE__ \
                << " : LHS='" << _sa << "' RHS='" << _sb << "'\n"; \
      std::abort(); \
    } \
  } while (0)

#define ASSERT_STR_NE(a,b) \
  do { \
    std::string _sa(a); std::string _sb(b); \
    if (!(_sa != _sb)) { \
      std::cerr << "[ASSERT_STR_NE failed] " << __FILE__ << ":" << __LINE__ \
                << " : both='" << _sa << "'\n"; \
      std::abort(); \
    } \
  } while (0)
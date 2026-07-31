#include "test/assert.hpp"

#include <kernel/core/checked.hpp>

namespace {

using rund::kernel::u64;
using rund::kernel::checked::add;
using rund::kernel::checked::ceil;
using rund::kernel::checked::mul;
using rund::kernel::checked::sub;

constexpr u64 kMax = ~u64{0u};

static_assert(add(0u, 0u));
static_assert(add(kMax, 0u));
static_assert(!add(kMax, 1u));
static_assert(mul(kMax, 0u));
static_assert(mul(kMax, 1u));
static_assert(!mul(kMax, 2u));
static_assert(ceil(0u, 0u) == 0u);
static_assert(ceil(1u, 0u) == 0u);
static_assert(ceil(0u, 7u) == 0u);
static_assert(ceil(1u, 7u) == 1u);
static_assert(ceil(kMax, 2u) == kMax / 2u + 1u);

int Addition() {
  u64 value = 41u;
  TEST_ASSERT(add(kMax, 0u, value));
  TEST_ASSERT(value == kMax);
  TEST_ASSERT(!add(kMax, 1u, value));
  TEST_ASSERT(value == kMax);
  TEST_ASSERT(!add(kMax, 1u));
  return 0;
}

int Subtraction() {
  u64 value = 47u;
  TEST_ASSERT(sub(kMax, kMax, value));
  TEST_ASSERT(value == 0u);
  TEST_ASSERT(sub(kMax, 1u, value));
  TEST_ASSERT(value == kMax - 1u);
  TEST_ASSERT(!sub(0u, 1u, value));
  TEST_ASSERT(value == kMax - 1u);
  return 0;
}

int Multiplication() {
  u64 value = 43u;
  TEST_ASSERT(mul(kMax, 1u, value));
  TEST_ASSERT(value == kMax);
  TEST_ASSERT(!mul(kMax, 2u, value));
  TEST_ASSERT(value == kMax);
  TEST_ASSERT(mul(kMax, 0u, value));
  TEST_ASSERT(value == 0u);
  TEST_ASSERT(!mul(kMax, 2u));

  value = 53u;
  TEST_ASSERT(mul(2u, 3u, 4u, value));
  TEST_ASSERT(value == 24u);
  TEST_ASSERT(!mul(kMax, 2u, 0u, value));
  TEST_ASSERT(value == 24u);
  TEST_ASSERT(!mul(kMax, 1u, 2u, value));
  TEST_ASSERT(value == 24u);
  return 0;
}

int Ceiling() {
  TEST_ASSERT(ceil(7u, 7u) == 1u);
  TEST_ASSERT(ceil(8u, 7u) == 2u);
  TEST_ASSERT(ceil(kMax, kMax) == 1u);
  TEST_ASSERT(ceil(kMax, 2u) == kMax / 2u + 1u);
  return 0;
}

} // namespace

int RunCheckedContract() {
  if (const int result = Addition(); result != 0) {
    return result;
  }
  if (const int result = Subtraction(); result != 0) {
    return result;
  }
  if (const int result = Multiplication(); result != 0) {
    return result;
  }
  return Ceiling();
}

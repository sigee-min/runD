#include <rund/counter.hpp>

#include "test/assert.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

constexpr std::uint64_t Maximum = std::numeric_limits<std::uint64_t>::max();
constexpr std::array<std::uint64_t, 7u> Values{
    0u, 1u, 2u, Maximum / 2u, Maximum - 2u, Maximum - 1u, Maximum};

[[nodiscard]] consteval bool AddLawsHold() {
  using rund::detail::counter::SaturatingAdd;
  for (const std::uint64_t left : Values) {
    if (SaturatingAdd(left, 0u) != left ||
        SaturatingAdd(left, Maximum) != Maximum) {
      return false;
    }
    for (const std::uint64_t right : Values) {
      if (SaturatingAdd(left, right) != SaturatingAdd(right, left)) {
        return false;
      }
      for (const std::uint64_t third : Values) {
        if (SaturatingAdd(SaturatingAdd(left, right), third) !=
            SaturatingAdd(left, SaturatingAdd(right, third))) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] consteval bool MultiplyLawsHold() {
  using rund::detail::counter::SaturatingMultiply;
  for (const std::uint64_t left : Values) {
    if (SaturatingMultiply(left, 0u) != 0u ||
        SaturatingMultiply(left, 1u) != left ||
        SaturatingMultiply(left, Maximum) != (left == 0u ? 0u : Maximum)) {
      return false;
    }
    for (const std::uint64_t right : Values) {
      if (SaturatingMultiply(left, right) != SaturatingMultiply(right, left)) {
        return false;
      }
      for (const std::uint64_t third : Values) {
        if (SaturatingMultiply(SaturatingMultiply(left, right), third) !=
            SaturatingMultiply(left, SaturatingMultiply(right, third))) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] consteval bool ReleaseLawsHold() {
  using rund::detail::counter::Delta;
  using rund::detail::counter::Remaining;
  for (const std::uint64_t current : Values) {
    for (const std::uint64_t released : Values) {
      const std::uint64_t expected =
          current == Maximum
              ? Maximum
              : released > current ? 0u : current - released;
      if (Remaining(current, released) != expected) {
        return false;
      }
    }
  }
  for (const std::uint64_t before : Values) {
    for (const std::uint64_t after : Values) {
      const std::uint64_t expected =
          before == Maximum || after == Maximum
              ? Maximum
              : after >= before ? after - before : 0u;
      if (Delta(before, after) != expected) {
        return false;
      }
    }
  }
  return true;
}

static_assert(noexcept(rund::detail::counter::SaturatingAdd(0u, 0u)));
static_assert(noexcept(rund::detail::counter::Accumulate(
    std::declval<std::uint64_t &>(), std::uint64_t{})));
static_assert(noexcept(rund::detail::counter::Remaining(0u, 0u)));
static_assert(noexcept(rund::detail::counter::Release(
    std::declval<std::uint64_t &>(), std::uint64_t{})));
static_assert(noexcept(rund::detail::counter::Delta(0u, 0u)));
static_assert(noexcept(rund::detail::counter::SaturatingMultiply(0u, 0u)));
static_assert(AddLawsHold());
static_assert(ReleaseLawsHold());
static_assert(MultiplyLawsHold());

} // namespace

int RunCounterContract() {
  using rund::detail::counter::SaturatingAdd;
  TEST_ASSERT(SaturatingAdd(0u, 0u) == 0u);
  TEST_ASSERT(SaturatingAdd(Maximum - 1u, 1u) == Maximum);
  TEST_ASSERT(SaturatingAdd(Maximum - 1u, 2u) == Maximum);
  TEST_ASSERT(SaturatingAdd(Maximum, Maximum) == Maximum);
  std::uint64_t total = Maximum - 1u;
  rund::detail::counter::Accumulate(total, 2u);
  TEST_ASSERT(total == Maximum);
  rund::detail::counter::Release(total, 4u);
  TEST_ASSERT(total == Maximum);
  TEST_ASSERT(rund::detail::counter::Remaining(9u, 4u) == 5u);
  TEST_ASSERT(rund::detail::counter::Remaining(3u, 4u) == 0u);
  TEST_ASSERT(rund::detail::counter::Delta(7u, 12u) == 5u);
  TEST_ASSERT(rund::detail::counter::Delta(Maximum - 1u, Maximum) == Maximum);
  TEST_ASSERT(rund::detail::counter::SaturatingMultiply(Maximum / 2u, 2u) ==
              Maximum - 1u);
  TEST_ASSERT(rund::detail::counter::SaturatingMultiply(Maximum, 2u) ==
              Maximum);
  return 0;
}

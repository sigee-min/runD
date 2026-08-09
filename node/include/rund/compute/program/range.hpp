#pragma once

#include <rund/compute/graph/info.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::compute {

enum class RangeKind : std::uint8_t {
  Direct,
  Shared,
  Prefix,
  Block,
};

// One capability-planned Range execution embedded in a compiled Program.
// The row is physical execution evidence, not a second planning surface.
struct RangeInfo final {
  std::uint32_t node{};
  RangeKind kind{RangeKind::Direct};
  std::uint32_t width{};
  std::uint32_t stages{};
  std::uint32_t shared_capacity{};
  std::uint64_t scratch_bytes{};
  graph::Fingerprint source{};
  graph::Fingerprint execution{};
};

struct RangeSnapshot final {
  std::size_t written{};
  std::size_t total{};

  [[nodiscard]] constexpr bool truncated() const noexcept {
    return written < total;
  }
};

static_assert(std::is_trivially_copyable_v<RangeInfo>);
static_assert(std::is_trivially_copyable_v<RangeSnapshot>);

} // namespace rund::compute

#pragma once

#include <cstdint>
#include <type_traits>

namespace rund::net::ready {

struct Set {
  std::uint64_t id = 0u;
  std::uint64_t generation = 0u;
};

static_assert(sizeof(Set) == 16u);
static_assert(std::is_standard_layout_v<Set>);
static_assert(std::is_trivially_copyable_v<Set>);

} // namespace rund::net::ready

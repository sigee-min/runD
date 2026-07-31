#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

inline constexpr std::uint32_t kScatterReduceWidth = 256u;

struct ScatterReduceParams final {
  std::uint64_t element_count{};
  std::uint64_t output_count{};
  std::uint32_t count_source{};
  std::uint32_t reserved{};
  std::uint32_t value_base{};
  std::uint32_t index_base{};
  std::uint32_t count_base{};
  std::uint32_t output_base{};
};

static_assert(sizeof(ScatterReduceParams) == 40u);

} // namespace rund::node::accel::detail

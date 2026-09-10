#pragma once

#include <rund/compute/fixed.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::detail {

struct GraphArg final {
  std::uint32_t value{};
  Type type{Type::I32};
  std::size_t count{};
  FixedFormat fixed_format{};
};
struct GraphOut final {
  std::uint32_t value{};
  Type type{Type::I32};
  std::size_t count{};
  FixedFormat fixed_format{};
  std::vector<GraphArg> outputs;
};

} // namespace rund::compute::detail
